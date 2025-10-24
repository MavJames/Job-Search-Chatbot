import streamlit as st
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
import threading
import tempfile
import uuid

# Optional: page config
st.set_page_config(page_title="Job Search Chatbot", layout="wide")

# -----------------------------
# Embeddings config
# -----------------------------
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

# -----------------------------
# Server capabilities + system prompt
# -----------------------------
def load_server_capabilities():
    """Load server capabilities from JSON file"""
    capabilities_path = Path(__file__).resolve().parent / "../Servers/server_capabilities.json"
    try:
        with open(capabilities_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Server capabilities file not found: {capabilities_path}")
        return {}

def build_enhanced_system_prompt():
    """Build system prompt incorporating server capabilities file"""
    capabilities = load_server_capabilities()

    base_prompt = """You are an expert Job Search engine, connecting to different servers to provide job searchers with the
    best opportunities for them.

AVAILABLE TOOLS AND THEIR CAPABILITIES:
"""
    for server_name, config in capabilities.items():
        base_prompt += f"""
**{server_name.upper()}**
Description: {config.get('description', 'No description available')}

Key Capabilities:
"""
        for capability in config.get('capabilities', []):
            base_prompt += f"  • {capability}\n"
        base_prompt += f"\nBest used for:\n"
        for use_case in config.get('use_cases', []):
            base_prompt += f"  • {use_case}\n"
        base_prompt += "\n"

    base_prompt += ""
    return base_prompt

SYSTEM_PROMPT = build_enhanced_system_prompt()

# -----------------------------
# Logging / env / config
# -----------------------------
logging.basicConfig(level=logging.INFO)
load_dotenv()
CONFIG_PATH = Path(__file__).resolve().parent / "../Servers/servers_config.json"

# -----------------------------
# App Title
# -----------------------------
st.title("Apiture Multi-Platform Chatbot")

# Keep last N turns (turn = user+assistant)
WINDOW_TURNS = 3

# -----------------------------
# LLM builders
# -----------------------------
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

# -----------------------------
# Event loop (persistent)
# -----------------------------
@st.cache_resource
def get_event_loop():
    """Create and return a persistent event loop for async operations"""
    loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(loop)
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    return loop

# -----------------------------
# Memory (LangMem)
# -----------------------------
@st.cache_resource
def initialize_memory_system():
    index_cfg = get_embedding_index()
    store = InMemoryStore(index=index_cfg)
    namespace = ("agent_memories",)
    # Import langmem lazily because version mismatches in langgraph/langmem can
    # raise ImportError at import time (see issue with CONFIG_KEY_STORE). If
    # langmem is not available or incompatible, we return no memory tools but
    # keep the app running.
    try:
        from langmem import create_manage_memory_tool, create_search_memory_tool

        memory_tools = [
            create_manage_memory_tool(namespace, store=store),
            create_search_memory_tool(namespace, store=store),
        ]
    except Exception as e:  # ImportError or runtime errors from incompatible packages
        logging.warning(
            f"langmem not available or failed to import; memory features disabled: {e}"
        )
        memory_tools = []

    return store, memory_tools

# -----------------------------
# Agent initialization
# -----------------------------
@st.cache_resource
def initialize_agent():
    """Initialize the agent with proper async handling"""
    loop = get_event_loop()

    async def _create_agent():
        try:
            with open(CONFIG_PATH) as f:
                config = json.load(f)
            servers = config.get("mcpServers", config.get("servers", {}))
            if not servers:
                raise ValueError("No servers found in configuration")

            client = MultiServerMCPClient(servers)

            all_tools = []
            sessions = {}

            for server_name in servers.keys():
                try:
                    session_ctx = client.session(server_name)
                    session = await session_ctx.__aenter__()
                    sessions[server_name] = (session, session_ctx)

                    tools = await load_mcp_tools(session)
                    all_tools.extend(tools)
                    logging.info(f"Connected to server: {server_name}, loaded {len(tools)} tools")
                except Exception as e:
                    logging.error(f"Failed to connect to server {server_name}: {e}")
                    continue

            if not all_tools:
                raise ValueError("No tools were loaded from any server")

            store, memory_tools = initialize_memory_system()
            all_tools.extend(memory_tools)

            llm = build_llm()

            agent = create_react_agent(
                llm,
                all_tools,
                store=store
            )
            logging.info(f"Agent initialized successfully with {len(all_tools)} tools and LangMem memory")
            return agent, client, sessions, store

        except Exception as e:
            logging.error(f"Failed to initialize agent: {e}")
            raise

    future = asyncio.run_coroutine_threadsafe(_create_agent(), loop)
    return future.result(timeout=120)

# -----------------------------
# Agent invocation helper
# -----------------------------
def run_agent_sync(messages, agent):
    loop = get_event_loop()

    async def _run_agent():
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

    future = asyncio.run_coroutine_threadsafe(_run_agent(), loop)
    return future.result(timeout=120)

# -----------------------------
# Initialize agent
# -----------------------------
try:
    with st.spinner("Initializing agent and connecting to servers..."):
        agent, client, sessions, store = initialize_agent()
except Exception as e:
    st.error(f"Failed to initialize agent: {str(e)}")
    st.error("Please check your configuration and environment variables.")
    st.stop()

# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_csv_path" not in st.session_state:
    st.session_state.uploaded_csv_path = None

# -----------------------------
# CSV uploader + preview + save to absolute path
# -----------------------------

uploaded_csv = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_csv:
    import pandas as pd  # Local import is fine; pandas already in deps
    # Preview (small sample for speed)
    try:
        preview_df = pd.read_csv(uploaded_csv, nrows=2000)
        st.caption(f"Preview of {uploaded_csv.name}")
        st.dataframe(preview_df.head(50), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV preview: {e}")

    # Persist to an absolute path so external MCP processes can read it
    tmp_dir = Path(tempfile.gettempdir())
    safe_name = f"{uuid.uuid4()}_{uploaded_csv.name}"
    csv_abs_path = tmp_dir / safe_name
    with open(csv_abs_path, "wb") as f:
        f.write(uploaded_csv.getbuffer())

    st.success(f"Saved CSV to: {csv_abs_path}")
    st.session_state.uploaded_csv_path = str(csv_abs_path)

# -----------------------------
# Helper: automatically include CSV path in prompts
# -----------------------------
def _with_csv_context(text: str) -> str:
    p = st.session_state.get("uploaded_csv_path")
    if p:
        return f"{text}\n\n(Uploaded CSV absolute path: {p})"
    return text

# -----------------------------
# Chat input + agent call
# -----------------------------
user_input = st.chat_input("Ask something...")
if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": _with_csv_context(user_input)}
    )

    msgs = st.session_state.chat_history
    window_size = max(2 * WINDOW_TURNS, 2)
    window = msgs[-window_size:]

    with st.spinner("Processing..."):
        try:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + window
            reply = run_agent_sync(messages, agent)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            err = f"Error processing request: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": err})

# -----------------------------
# Render conversation
# -----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
