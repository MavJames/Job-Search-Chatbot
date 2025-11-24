import asyncio
import json
import logging
import os
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import docx
import PyPDF2
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from system_prompt import build_enhanced_system_prompt

load_dotenv()
logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class AppState:
    def __init__(self) -> None:
        self.agent = None
        self.client: MultiServerMCPClient | None = None
        self.sessions: dict[str, Any] = {}
        self.exit_stack: AsyncExitStack | None = None
        self.store: InMemoryStore | None = None
        self.runtime_metadata: dict[str, Any] | None = None
        self.llm = None
        self.tools: list[Any] = []
        self.lock = asyncio.Lock()


state = AppState()


def get_embedding_index() -> dict[str, Any]:
    """Configure embeddings for the agent memory store."""
    azure_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    ]
    if all(os.getenv(var) for var in azure_vars):
        return {
            "dims": 1536,
            "embed": f"azure_openai:{os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT')}",
        }

    if os.getenv("OPENAI_API_KEY"):
        return {
            "dims": 1536,
            "embed": "openai:text-embedding-3-small",
        }

    raise ValueError(
        "Missing both Azure and OpenAI embedding configuration. Need either "
        "AZURE_OPENAI_* vars or OPENAI_API_KEY",
    )


def get_llm():
    """Build and return the language model instance."""
    azure_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    ]
    if all(os.getenv(var) for var in azure_vars):
        return AzureChatOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            temperature=0.1,
            max_tokens=1500,
            timeout=30,
            streaming=True,
        )

    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.1,
            max_tokens=1500,
            timeout=30,
            streaming=True,
        )

    raise ValueError(
        "Missing both Azure and OpenAI configuration. Need either "
        "AZURE_OPENAI_* vars or OPENAI_API_KEY",
    )


async def initialize_agent_resources(exit_stack: AsyncExitStack) -> None:
    """Initialize the agent, MCP sessions, memory store, and metadata."""
    async with state.lock:
        if state.agent:
            return

        logging.info("Initializing agent resources...")
        state.exit_stack = exit_stack
        state.sessions = {}
        state.tools = []
        state.runtime_metadata = None

        try:
            state.llm = get_llm()

            config_path = Path(__file__).resolve().parent.parent.parent / "Servers/servers_config.json"
            with open(config_path) as config_file:
                config = json.load(config_file)

            servers = config.get("mcpServers", {})
            if not servers:
                raise ValueError("No MCP servers found in configuration.")

            state.client = MultiServerMCPClient(servers)
            exit_stack.push_async_callback(state.client.aclose)

            server_info: dict[str, dict[str, Any]] = {}

            for name in servers.keys():
                try:
                    session = await exit_stack.enter_async_context(state.client.session(name))
                    state.sessions[name] = session

                    tools = await load_mcp_tools(session)
                    state.tools.extend(tools)
                    server_info[name] = {"tools": len(tools), "status": "Connected"}
                    logging.info("Connected to '%s' and loaded %d tools.", name, len(tools))
                except Exception as exc:
                    server_info[name] = {"tools": 0, "status": "Error"}
                    logging.error("Failed to connect to server '%s': %s", name, exc)

            if not state.tools:
                raise ValueError("No tools were loaded from any server.")

            index_cfg = get_embedding_index()
            state.store = InMemoryStore(index=index_cfg)
            namespace = ("agent_memories",)
            memory_tools = [
                create_manage_memory_tool(namespace, store=state.store),
                create_search_memory_tool(namespace, store=state.store),
            ]
            state.tools.extend(memory_tools)

            state.agent = create_react_agent(state.llm, state.tools, store=state.store)

            state.runtime_metadata = {
                "servers": server_info,
                "summary": {
                    "total_tools": len(state.tools),
                    "memory_tools": len(memory_tools),
                    "connected_servers": sum(
                        1 for info in server_info.values() if info["status"] == "Connected"
                    ),
                },
            }
            logging.info("Agent resources initialized successfully.")

        except Exception as exc:
            logging.error("Fatal error during agent initialization: %s", exc)
            await exit_stack.aclose()
            state.exit_stack = None
            state.client = None
            state.sessions.clear()
            state.tools.clear()
            state.agent = None
            state.store = None
            state.runtime_metadata = None
            state.llm = None
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        await initialize_agent_resources(stack)
        yield
    finally:
        logging.info("Shutting down server sessions...")
        try:
            await stack.aclose()
        finally:
            state.exit_stack = None
            state.sessions.clear()
            state.tools.clear()
            state.agent = None
            state.store = None
            state.client = None
            state.runtime_metadata = None
            state.llm = None


app = FastAPI(title="Job Search Assistant Backend", lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    resume_text: str | None = None
    persona: str | None = None


@app.get("/health")
def health_check():
    """Health check endpoint for container orchestrators."""
    return {"status": "ok"}


@app.post("/initialize")
async def get_initialization_status():
    """Provide the status of agent initialization to the UI."""
    if state.agent and state.runtime_metadata:
        return {"status": "Initialized successfully", "metadata": state.runtime_metadata}

    for _ in range(10):
        if state.agent:
            return {"status": "Initialized successfully", "metadata": state.runtime_metadata}
        await asyncio.sleep(1)

    raise HTTPException(
        status_code=503,
        detail="Agent initialization is taking longer than expected. Please try again shortly.",
    )

@app.post("/chat/stream")
async def stream_chat(chat_request: ChatRequest):
    """Handle chat requests and stream the agent's response."""
    if not state.agent:
        raise HTTPException(status_code=503, detail="Agent is not initialized.")

    system_prompt = build_enhanced_system_prompt(chat_request.resume_text, chat_request.persona)

    messages = [SystemMessage(content=system_prompt)]
    for msg in chat_request.messages:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))

    def make_json_safe(value):
        """Convert LangChain objects to JSON-serializable structures."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {k: make_json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [make_json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            return make_json_safe(value.model_dump())
        if hasattr(value, "dict"):
            return make_json_safe(value.dict())
        if hasattr(value, "content"):
            return make_json_safe(getattr(value, "content"))
        return str(value)

    async def event_generator():
        def extract_chunk_text(chunk) -> str:
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if text:
                            parts.append(text)
                    elif hasattr(item, "text"):
                        text = getattr(item, "text")
                        if text:
                            parts.append(text)
                return "".join(parts)
            return ""

        collected_tokens: list[str] = []

        try:
            config = {"recursion_limit": 20, "configurable": {"thread_id": "streamlit-session"}}
            async for event in state.agent.astream_events(
                {"messages": messages},
                config=config,
                version="v1",
            ):
                event_type = event.get("event")

                if event_type == "on_tool_start":
                    payload = {
                        "type": "tool_start",
                        "name": event.get("name"),
                        "input": make_json_safe(event.get("data", {}).get("input")),
                    }
                    yield {"event": "tool_start", "data": json.dumps(payload)}

                elif event_type == "on_tool_end":
                    payload = {
                        "type": "tool_end",
                        "name": event.get("name"),
                        "output": make_json_safe(event.get("data", {}).get("output")),
                    }
                    yield {"event": "tool_end", "data": json.dumps(payload)}

                elif event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if not chunk:
                        continue
                    text = extract_chunk_text(chunk)
                    if not text:
                        continue
                    collected_tokens.append(text)
                    yield {"event": "message", "data": json.dumps({"type": "token", "content": text})}

                elif event_type == "on_chain_end":
                    final_text = "".join(collected_tokens).strip()
                    if final_text:
                        yield {"event": "message", "data": json.dumps({"type": "final", "content": final_text})}

            yield {"event": "end", "data": json.dumps({"type": "end"})}

        except Exception as exc:
            logging.error("Error during agent stream: %s", exc)
            payload = {"type": "error", "content": "An error occurred. Please try again."}
            yield {"event": "error", "data": json.dumps(payload)}

    return EventSourceResponse(event_generator())

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """Handle resume uploads securely."""
    if file.content_type not in [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF or DOCX file.")

    try:
        # Secure filename and save
        safe_filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())

        # Extract text
        if file.content_type == "application/pdf":
            text = extract_text_from_pdf(filepath)
        else:
            text = extract_text_from_docx(filepath)
            
        if not text:
            raise HTTPException(status_code=500, detail="Could not extract text from the resume.")
            
        return {"status": "success", "resume_text": text, "filename": safe_filename}

    except Exception as exc:
        logging.error("Error processing uploaded resume: %s", exc)
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {exc}")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        text = ""
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text.strip()
    except Exception as exc:
        logging.error("Error extracting PDF text: %s", exc)
        return None

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()
    except Exception as exc:
        logging.error("Error extracting DOCX text: %s", exc)
        return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)
