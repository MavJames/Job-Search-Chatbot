import asyncio
import logging
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel, Field

from Clients.config import get_servers_config, get_settings
from Clients.utils import get_event_loop


def get_embedding_index():
    """
    Configure LangMem/InMemoryStore to use Azure or OpenAI embeddings.
    Tries Azure first, falls back to OpenAI.
    """
    settings = get_settings()
    if (
        settings.azure_openai_api_key
        and settings.azure_openai_endpoint
        and settings.azure_openai_api_version
        and settings.azure_openai_embeddings_deployment
    ):
        return {
            "dims": 1536,  # matches text-embedding-3-small family
            "embed": f"azure_openai:{settings.azure_openai_embeddings_deployment}",
        }

    if settings.openai_api_key:
        return {
            "dims": 1536,  # matches text-embedding-3-small
            "embed": "openai:text-embedding-3-small",
        }

    raise ValueError(
        "Missing both Azure and OpenAI embedding configuration. "
        "Need either AZURE_OPENAI_* vars or OPENAI_API_KEY"
    )


def build_llm():
    """Builds the LLM for the agent, checking for Azure credentials first."""
    settings = get_settings()
    if (
        settings.azure_openai_api_key
        and settings.azure_openai_endpoint
        and settings.azure_openai_api_version
        and settings.azure_openai_deployment
    ):
        return AzureChatOpenAI(
            api_key=settings.azure_openai_api_key,
            azure_endpoint=settings.azure_openai_endpoint,
            api_version=settings.azure_openai_api_version,
            deployment_name=settings.azure_openai_deployment,
            temperature=0.1,
            max_tokens=1500,
            timeout=30,
        )

    if settings.openai_api_key:
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model="gpt-4o",
            temperature=0.1,
            max_tokens=1500,
            timeout=30,
        )

    raise ValueError(
        "Missing both Azure and OpenAI configuration. "
        "Need either AZURE_OPENAI_* vars or OPENAI_API_KEY"
    )


def initialize_memory_system():
    """Initializes the memory system for the agent."""
    index_cfg = get_embedding_index()
    store = InMemoryStore(index=index_cfg)
    namespace = ("agent_memories",)
    memory_tools = [
        create_manage_memory_tool(namespace, store=store),
        create_search_memory_tool(namespace, store=store),
    ]
    return store, memory_tools


def initialize_agent():
    """Initialize the agent with proper async handling"""
    loop = get_event_loop()
    settings = get_settings()

    async def _create_agent():
        try:
            config = get_servers_config()

            # Handle filesystem root path if it's configured
            filesystem_cfg = config.get("mcpServers", {}).get("filesystem", {})
            fs_args = filesystem_cfg.get("args") if filesystem_cfg else None
            if fs_args:
                if settings.filesystem_root:
                    fs_root = Path(settings.filesystem_root).expanduser().resolve()
                else:
                    config_path = (
                        Path(__file__).resolve().parent
                        / "../Servers/servers_config.json"
                    )
                    candidate = Path(fs_args[-1])
                    if not candidate.is_absolute():
                        candidate = (config_path.resolve().parent / candidate).resolve()
                    fs_root = candidate
                fs_args[-1] = str(fs_root)

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

                    # PATCH: Fix schema for filesystem tools if they are broken
                    if server_name == "filesystem":
                        for tool in tools:
                            if tool.name == "read_file" and (
                                not tool.args_schema
                                or not hasattr(tool.args_schema, "model_json_schema")
                            ):

                                class ReadFileArgs(BaseModel):
                                    path: str = Field(
                                        ...,
                                        description="The absolute path to the file to read",
                                    )

                                tool.args_schema = ReadFileArgs
                            elif tool.name == "write_file" and (
                                not tool.args_schema
                                or not hasattr(tool.args_schema, "model_json_schema")
                            ):

                                class WriteFileArgs(BaseModel):
                                    path: str = Field(
                                        ...,
                                        description="The absolute path to the file to write",
                                    )
                                    content: str = Field(
                                        ...,
                                        description="The content to write to the file",
                                    )

                                tool.args_schema = WriteFileArgs

                    all_tools.extend(tools)
                    logging.info(
                        f"Connected to server: {server_name}, loaded {len(tools)} tools"
                    )
                except Exception as e:
                    logging.error(f"Failed to connect to server {server_name}: {e}")
                    continue

            if not all_tools:
                raise ValueError("No tools were loaded from any server")

            store, memory_tools = initialize_memory_system()
            all_tools.extend(memory_tools)

            llm = build_llm()

            agent = create_react_agent(llm, all_tools, store=store)
            logging.info(
                f"Agent initialized successfully with {len(all_tools)} tools and LangMem memory"
            )
            return agent, client, sessions, store

        except Exception as e:
            logging.error(f"Failed to initialize agent: {e}")
            raise

    future = asyncio.run_coroutine_threadsafe(_create_agent(), loop)
    return future.result(timeout=120)


def run_agent_sync(messages, agent):
    """Lets streamlit use asynchronous servers, even though it is synchronous"""
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
            return (
                result["messages"][-1].content
                if result.get("messages")
                else "No reply."
            )
        except Exception as e:
            logging.exception("Agent error")
            return f"Error: {str(e)}"

    future = asyncio.run_coroutine_threadsafe(_run_agent(), loop)
    return future.result(timeout=120)
