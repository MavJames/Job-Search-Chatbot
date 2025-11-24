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
from langmem import create_manage_memory_tool, create_search_memory_tool
import threading
import tempfile
import uuid
import PyPDF2
import docx
from datetime import datetime

st.set_page_config(page_title="Job Search Assistant", layout="wide")

#Configures the embedding model to convert text to vectors for semantic search in the memory system
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

#Extracts information from file based on the type given
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

#Builds system prompt with the extrected text from the attached resume, NEEDS TO BE OPTIMIZED
def load_server_capabilities():
    """Load server capabilities from JSON file"""
    capabilities_path = Path(__file__).resolve().parent / "../Servers/server_capabilities.json"
    try:
        with open(capabilities_path) as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Server capabilities file not found: {capabilities_path}")
        return {}

def build_enhanced_system_prompt(resume_text=None):
    """Build system prompt incorporating server capabilities and resume context"""
    capabilities = load_server_capabilities()
    current_date = datetime.now().strftime("%B %d, %Y")

    base_prompt = f"""You are a Job Search Assistant helping candidates find opportunities and navigate applications. When asked to create a resume
    return a docx file that is in Microsoft Word format. Today is {current_date}.


    ## YOUR ROLE & PHILOSOPHY

You help candidates pursue their CAREER GOALS, not just roles matching their current experience. A candidate's current position (e.g., intern, entry-level) does NOT define what they're capable of or aspiring to achieve. Always ask about:
- What roles they WANT to pursue
- What industries or companies interest them
- What skills they want to use or develop
- Their career trajectory and goals

NEVER assume someone wants jobs similar to their current role. An intern may be seeking full-time positions, a data analyst may want to move into engineering, etc.

## WORKFLOW

### 1. DISCOVERY PHASE
Before searching for jobs, understand:
- What TYPE of role are they targeting? (e.g., "Data Scientist", "Software Engineer", "Product Manager")
- What LEVEL? (Intern, Entry-level, Mid-level, Senior)
- Preferred LOCATION or remote preference

Ask clarifying questions! Don't make assumptions.

### 2. JOB SEARCH PHASE
Use the job search tools to find positions matching their GOALS (not just experience):
- Search by their TARGET role title, not current title
- Consider various related titles (e.g., "Data Scientist", "ML Engineer", "Applied Scientist")
- Search across multiple locations if they're flexible
- Cast a wide net initially, then refine based on feedback

Present findings clearly:
- Job title and company
- Location and work arrangement
- Key requirements and responsibilities
- Why it matches their goals
- Any gaps or stretch requirements to address

### 3. APPLICATION STRATEGY PHASE
For jobs they want to apply to:
- Analyze the job description thoroughly
- Identify key requirements and desired qualifications
- Map their experience and skills to requirements
- Suggest how to position their background
- Note any skills to emphasize or gaps to address

### 4. DOCUMENT CREATION PHASE
When creating resumes or cover letters:

**RESUMES:**
- Tailor to the SPECIFIC job posting
- Lead with relevant skills and projects, not chronological history
- Quantify achievements wherever possible
- Highlight transferable skills from different contexts
- Position current/past roles in terms of relevant skills gained
- Use keywords from the job description naturally
- Format: Create as .docx (Microsoft Word format) using create_resume tool
- Keep to ONE page unless explicitly requested otherwise

**COVER LETTERS:**
- Address specific job requirements and company
- Tell the story of WHY they're pursuing this role
- Connect their background to the role's needs (even if indirect)
- Show genuine interest and research about the company
- Address any career transitions or non-traditional paths proactively
- 3-4 substantial paragraphs
- Professional, enthusiastic tone
- Format: Create as .docx using create_cover_letter tool

## KEY PRINCIPLES

1. **Goal-Oriented, Not Experience-Limited**: Help candidates reach for roles they ASPIRE to, not just what they've done
2. **Strategic Positioning**: Frame experience in terms of skills and impact relevant to target role
3. **Proactive Gap Addressing**: Help candidates address experience gaps confidently
4. **Customization is Key**: Every resume and cover letter should be tailored to the specific opportunity
5. **Realistic but Optimistic**: Be honest about stretches while encouraging qualified candidates
6. **Continuous Refinement**: Iterate on documents based on feedback

## COMMUNICATION STYLE

- Ask clarifying questions before taking action
- Explain your reasoning and suggestions
- Offer options when multiple approaches exist
- Be encouraging about career transitions and growth
- Use clear, professional language
- Confirm understanding before creating documents
"""

    if resume_text:
        base_prompt += f"""CANDIDATE RESUME:
{resume_text}

"""

    base_prompt += """AVAILABLE TOOLS:
"""
    for server_name, config in capabilities.items():
        base_prompt += f"\n{server_name}: {config.get('description', '')}\n"

    return base_prompt

#Logging to track flow, loading env, getting json file of servers, and titling the streamlit app
logging.basicConfig(level=logging.INFO)
load_dotenv()
CONFIG_PATH = Path(__file__).resolve().parent / "../Servers/servers_config.json"
st.title("Job Search Assistant")

#Keep last N turns (turn = user+assistant)
WINDOW_TURNS = 3

#Creates the brain of the agent by connecting to an llm that is provided in the env, checks for credentials Azure first then falls back to OpenAI
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

#Toughest part mcp servers run asynchronously but streamlit doesn't, this creates a separate thread running an async event loop
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

#Database to hold memories that are mainly prompted by the user and uses the embedding model
@st.cache_resource
def initialize_memory_system():
    index_cfg = get_embedding_index()
    store = InMemoryStore(index=index_cfg)
    namespace = ("agent_memories",)
    memory_tools = [
        create_manage_memory_tool(namespace, store=store),
        create_search_memory_tool(namespace, store=store),
    ]
    return store, memory_tools

#Initializing the agent: Reads the config file to see which servers to connect to, connects to the servers and collect the tools from them
#Adds memory tool, runs build_llm and creates the REACT agent from Langgraph
@st.cache_resource
def initialize_agent():
    """Initialize the agent with proper async handling"""
    loop = get_event_loop()

    async def _create_agent():
        try:
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

#Lets streamlit use asynchronous servers, even though it is synchronous 
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

#Runs initialize agent with the streamlit spinner to show the process is running
try:
    with st.spinner("Initializing agent and connecting to job search servers..."):
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
if "resume_path" not in st.session_state:
    st.session_state.resume_path = None
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

#Resume uploader for pdf or docx file formats and generates a uuid to avoid filename conflicts, remembers for entire session
with st.sidebar:
    st.image("/Users/carly.watkins/Desktop/Job-Search-Chatbot/Desktop/uncw_logo.png", width='stretch')
    st.header("Upload Resume")
    uploaded_resume = st.file_uploader("Choose your resume", type=["pdf", "docx", "doc"])

    if uploaded_resume:
        tmp_dir = Path(tempfile.gettempdir())
        safe_name = f"{uuid.uuid4()}_{uploaded_resume.name}"
        resume_abs_path = tmp_dir / safe_name
        
        with open(resume_abs_path, "wb") as f:
            f.write(uploaded_resume.getbuffer())

        resume_text = extract_resume_text(resume_abs_path)
        
        if resume_text:
            st.success("Resume uploaded")
            st.session_state.resume_path = str(resume_abs_path)
            st.session_state.resume_text = resume_text
        else:
            st.error("Could not extract text from resume")
            st.session_state.resume_path = None
            st.session_state.resume_text = None


# Build system prompt with resume context
def get_current_system_prompt():
    """Get system prompt with current resume context"""
    resume_text = st.session_state.get("resume_text")
    return build_enhanced_system_prompt(resume_text)

#User interface that takes text handles memory, sends input to agent and then saves reponse in history
user_input = st.chat_input("Ask about jobs...")

if user_input:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )

    msgs = st.session_state.chat_history
    window_size = max(2 * WINDOW_TURNS, 2)
    window = msgs[-window_size:]

    with st.spinner("Processing..."):
        try:
            system_prompt = get_current_system_prompt()
            messages = [{"role": "system", "content": system_prompt}] + window
            reply = run_agent_sync(messages, agent)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            err = f"Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": err})

#Shows past messages in session
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
