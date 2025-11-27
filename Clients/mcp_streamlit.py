import logging
import tempfile
import uuid
from pathlib import Path

import streamlit as st

try:
    from Clients.agent import initialize_agent, run_agent_sync
    from Clients.prompts import build_enhanced_system_prompt
    from Clients.utils import extract_resume_text
except ImportError:
    # Fallback if running from Clients directory
    from agent import initialize_agent, run_agent_sync
    from prompts import build_enhanced_system_prompt
    from utils import extract_resume_text

st.set_page_config(page_title="Job Search Assistant", layout="wide")

logging.basicConfig(level=logging.INFO)
st.title("Job Search Assistant")

# Keep last N turns (turn = user+assistant)
WINDOW_TURNS = 3

# Runs initialize agent with the streamlit spinner to show the process is running
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

# Resume uploader for pdf or docx file formats and generates a uuid to avoid filename conflicts, remembers for entire session
with st.sidebar:
    logo_path = Path(__file__).resolve().parent.parent / "uncw_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width="stretch")
    else:
        st.warning("Logo not found")
    st.header("Upload Resume")
    uploaded_resume = st.file_uploader(
        "Choose your resume", type=["pdf", "docx", "doc"]
    )

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


# User interface that takes text handles memory, sends input to agent and then saves reponse in history
user_input = st.chat_input("Ask about jobs...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    msgs = st.session_state.chat_history
    window_size = max(2 * WINDOW_TURNS, 2)
    window = msgs[-window_size:]

    with st.spinner("Processing..."):
        try:
            system_prompt = get_current_system_prompt()
            messages = [{"role": "system", "content": system_prompt}] + window
            reply = run_agent_sync(messages, agent)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": reply}
            )
        except Exception as e:
            err = f"Error: {str(e)}"
            st.session_state.chat_history.append({"role": "assistant", "content": err})

# Shows past messages in session
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
