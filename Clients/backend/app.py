
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import os
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
import threading
import PyPDF2
import docx
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from system_prompt import build_enhanced_system_prompt

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for agent and related components
agent = None
client = None
sessions = None
store = None
runtime_metadata = None
loop = None

# --- Helper Functions from mcp_streamlit.py ---

def get_embedding_index():
    """
    Configure LangMem/InMemoryStore to use Azure or OpenAI embeddings.
    Tries Azure first, falls back to OpenAI.
    """
    azure_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT",
    ]
    if all(os.getenv(var) for var in azure_vars):
        return {
            "dims": 1536,  # matches text-embedding-3-small family
            "embed": f"azure_openai:{os.getenv('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT')}",
        }

    if os.getenv("OPENAI_API_KEY"):
        return {
            "dims": 1536,  # matches text-embedding-3-small
            "embed": "openai:text-embedding-3-small",
        }

    raise ValueError(
        f"Missing both Azure and OpenAI embedding configuration. Need either: "
        f"{', '.join(azure_vars)} or OPENAI_API_KEY"
    )

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(docx_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        return None

def extract_resume_text(file_path):
    """Extract text from resume file (PDF or DOCX)"""
    file_path = Path(file_path)
    if file_path.suffix.lower() == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_path.suffix.lower() in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        return None



def build_llm():
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
        )

    if os.getenv("OPENAI_API_KEY"):
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1500,
            timeout=30,
        )

    raise ValueError(
        "Missing both Azure and OpenAI configuration. Need either "
        "AZURE_OPENAI_* vars or OPENAI_API_KEY"
    )

def get_event_loop():
    """Create and return a persistent event loop for async operations"""
    global loop
    if loop is None:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=run_loop, args=(loop,), daemon=True)
        thread.start()
    return loop

def run_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

def initialize_memory_system():
    index_cfg = get_embedding_index()
    store = InMemoryStore(index=index_cfg)
    namespace = ("agent_memories",)
    memory_tools = [
        create_manage_memory_tool(namespace, store=store),
        create_search_memory_tool(namespace, store=store),
    ]
    return store, memory_tools

async def _create_agent():
    global agent, client, sessions, store, runtime_metadata
    try:
        CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "Servers/servers_config.json"
        with open(CONFIG_PATH) as f:
            config = json.load(f)

        filesystem_cfg = config.get("mcpServers", {}).get("filesystem", {})
        fs_args = filesystem_cfg.get("args") if filesystem_cfg else None
        if fs_args:
            env_root = os.getenv("FILESYSTEM_ROOT")
            if env_root:
                fs_root = Path(env_root).expanduser().resolve()
            else:
                candidate = Path(fs_args[-1])
                if not candidate.is_absolute():
                    candidate = (CONFIG_PATH.resolve().parent / candidate).resolve()
                fs_root = candidate
            fs_args[-1] = str(fs_root)

        servers = config.get("mcpServers", config.get("servers", {}))
        if not servers:
            raise ValueError("No servers found in configuration")

        client = MultiServerMCPClient(servers)

        all_tools = []
        sessions = {}
        server_info = {}

        for server_name in servers.keys():
            try:
                session_ctx = client.session(server_name)
                session = await session_ctx.__aenter__()
                sessions[server_name] = (session, session_ctx)

                tools = await load_mcp_tools(session)
                all_tools.extend(tools)
                server_info[server_name] = {
                    "tools": len(tools),
                    "status": "Connected"
                }
                logging.info(f"Connected to server: {server_name}, loaded {len(tools)} tools")
            except Exception as e:
                logging.error(f"Failed to connect to server {server_name}: {e}")
                server_info[server_name] = {
                    "tools": 0,
                    "status": "Error"
                }
                continue

        if not all_tools:
            raise ValueError("No tools were loaded from any server")

        store, memory_tools = initialize_memory_system()
        all_tools.extend(memory_tools)

        summary = {
            "total_tools": len(all_tools),
            "memory_tools": len(memory_tools),
            "connected_servers": sum(1 for info in server_info.values() if info.get("status") == "Connected"),
        }
        runtime_metadata = {"servers": server_info, "summary": summary}

        llm = build_llm()

        agent = create_react_agent(
            llm,
            all_tools,
            store=store
        )
        logging.info(f"Agent initialized successfully with {len(all_tools)} tools and LangMem memory")
        return agent, client, sessions, store, runtime_metadata

    except Exception as e:
        logging.error(f"Failed to initialize agent: {e}")
        raise

async def _run_agent(messages):
    try:
        result = await agent.ainvoke(
            {"messages": messages},
            {
                "recursion_limit": 20,
                "configurable": {"thread_id": "streamlit-session"},
            },
        )
        return result["messages"][-1].content if result.get("messages") else "No reply."
    except Exception as e:
        logging.exception("Agent error")
        return f"Error: {str(e)}"

# --- Flask Endpoints ---

@app.route('/initialize', methods=['POST'])
def initialize():
    global agent, client, sessions, store, runtime_metadata, loop
    if agent:
        return jsonify({"status": "Already initialized", "metadata": runtime_metadata})

    try:
        loop = get_event_loop()
        future = asyncio.run_coroutine_threadsafe(_create_agent(), loop)
        agent, client, sessions, store, runtime_metadata = future.result(timeout=120)
        return jsonify({"status": "Initialized successfully", "metadata": runtime_metadata})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/run_agent', methods=['POST'])
def run_agent():
    if not agent:
        return jsonify({"status": "error", "message": "Agent not initialized"}), 500

    data = request.json
    messages = data.get('messages')
    
    if not messages:
        return jsonify({"status": "error", "message": "No messages provided"}), 400

    try:
        future = asyncio.run_coroutine_threadsafe(_run_agent(messages), loop)
        reply = future.result(timeout=120)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        resume_text = extract_resume_text(filepath)
        if resume_text:
            return jsonify({"status": "success", "resume_text": resume_text, "filepath": filepath})
        else:
            return jsonify({"status": "error", "message": "Could not extract text from resume"}), 500

@app.route('/system_prompt', methods=['POST'])
def get_system_prompt():
    data = request.json
    resume_text = data.get('resume_text')
    prompt = build_enhanced_system_prompt(resume_text)
    return jsonify({"system_prompt": prompt})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    app.run(host='0.0.0.0', port=5001, debug=True)
