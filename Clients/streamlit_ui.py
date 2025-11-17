
import streamlit as st
import requests
import os

st.set_page_config(page_title="Job Search Assistant", layout="wide")

# --- UI Styling ---
THEME_CSS = """
<style>
.stApp {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2.5rem;
}
.hero-card {
    background: linear-gradient(120deg, #1f2937 0%, #2563eb 100%);
    color: #ffffff;
    padding: 1.5rem 1.75rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
}
.hero-title {
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.hero-subtitle {
    font-size: 0.95rem;
    opacity: 0.92;
    margin: 0;
}
.metric-row {
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
}
.metric-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    margin-bottom: 0.35rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #111827;
}
.metric-helper {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.35rem;
}
.sidebar-section-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.sidebar-badge {
    background: #e0e7ff;
    border-radius: 999px;
    display: inline-block;
    padding: 0.35rem 0.85rem;
    margin-right: 0.35rem;
    margin-bottom: 0.4rem;
    font-size: 0.85rem;
    color: #1e1b4b;
}
.sidebar-footnote {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-bottom: 1rem;
}
[data-testid="stChatMessage"] {
    margin-bottom: 1rem;
}
[data-testid="stChatMessageUser"] > div {
    background: #1d4ed8;
    color: #ffffff;
    border-radius: 18px;
    padding: 1rem;
    box-shadow: 0 18px 40px rgba(37, 99, 235, 0.25);
}
[data-testid="stChatMessageAssistant"] > div {
    background: #ffffff;
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
}
.info-expander p {
    margin-bottom: 0.25rem;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)

# --- Backend API Communication ---
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001")

def initialize_backend():
    try:
        response = requests.post(f"{BACKEND_URL}/initialize")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to backend: {e}")
        return None

def run_agent_on_backend(messages):
    try:
        response = requests.post(f"{BACKEND_URL}/run_agent", json={"messages": messages})
        response.raise_for_status()
        return response.json().get("reply", "No reply from agent.")
    except requests.exceptions.RequestException as e:
        return f"Error communicating with agent: {e}"

def upload_resume_to_backend(uploaded_file):
    try:
        files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        response = requests.post(f"{BACKEND_URL}/upload_resume", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to upload resume: {e}")
        return None

def get_system_prompt_from_backend(resume_text=None):
    try:
        response = requests.post(f"{BACKEND_URL}/system_prompt", json={"resume_text": resume_text})
        response.raise_for_status()
        return response.json().get("system_prompt")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get system prompt: {e}")
        return None

# --- Main App Logic ---

st.title("Job Search Assistant")

# Initialize backend on first run
if "backend_initialized" not in st.session_state:
    with st.spinner("Initializing agent and connecting to job search servers..."):
        init_data = initialize_backend()
        if init_data and init_data.get("status") == "Initialized successfully":
            st.session_state.backend_initialized = True
            st.session_state.ui_metrics = init_data.get("metadata", {"servers": {}, "summary": {}})
            st.rerun()
        elif init_data:
            st.error(f"Backend initialization failed: {init_data.get('message')}")
            st.stop()
        else:
            st.error("Could not connect to the backend. Please ensure it is running.")
            st.stop()

# Session state management
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

# --- UI Components ---

ui_metrics = st.session_state.get("ui_metrics", {"servers": {}, "summary": {}})
server_info = ui_metrics.get("servers", {})
summary_info = ui_metrics.get("summary", {})

assistant_turns = sum(1 for msg in st.session_state.chat_history if msg.get("role") == "assistant")
total_messages = len(st.session_state.chat_history)
resume_ready = bool(st.session_state.resume_text)

hero_html = (
    "<div class='hero-card'>"
    "<div class='hero-title'>Job Search Copilot</div>"
    "<p class='hero-subtitle'>Blend MCP tooling, personalized memory, and GPT intelligence to move from discovery to tailored applications in minutes.</p>"
    "</div>"
)
st.markdown(hero_html, unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.markdown(
    f"<div class='metric-card'><div class='metric-label'>Assistant replies</div>"
    f"<div class='metric-value'>{assistant_turns}</div>"
    f"<div class='metric-helper'>Across {total_messages} total messages</div></div>",
    unsafe_allow_html=True,
)
connected_servers = summary_info.get("connected_servers", 0)
total_tools = summary_info.get("total_tools", 0)
col2.markdown(
    f"<div class='metric-card'><div class='metric-label'>Active tooling</div>"
    f"<div class='metric-value'>{connected_servers}</div>"
    f"<div class='metric-helper'>{total_tools} tools available</div></div>",
    unsafe_allow_html=True,
)
resume_status = "Ready" if resume_ready else "Pending"
resume_helper = "Resume context synced" if resume_ready else "Upload to unlock tailoring"
col3.markdown(
    f"<div class='metric-card'><div class='metric-label'>Resume status</div>"
    f"<div class='metric-value'>{resume_status}</div>"
    f"<div class='metric-helper'>{resume_helper}</div></div>",
    unsafe_allow_html=True,
)

with st.expander("Assistant playbook", expanded=False):
    st.markdown(
        "- Start with your target role, locations, or companies so the agent can focus the search.\n"
        "- Ask for salary insights or top companies to prioritize opportunities quickly.\n"
        "- When you find a posting you like, request a tailored resume or cover letter and download the generated DOCX.\n"
        "- Re-upload a revised resume any time to refresh the assistant's memory context.")

# Sidebar for resume upload and status
with st.sidebar:
    st.markdown("<div class='sidebar-section-title'>Session status</div>", unsafe_allow_html=True)
    if server_info:
        for name, info in server_info.items():
            status_label = info.get("status", "Unknown")
            tools_label = info.get("tools", 0)
            badge_text = f"{name} · {status_label.lower()}"
            if tools_label:
                badge_text += f" · {tools_label} tools"
            st.markdown(f"<div class='sidebar-badge'>{badge_text}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='sidebar-badge'>No MCP servers detected</div>", unsafe_allow_html=True)

    if summary_info:
        st.markdown(
            f"<div class='sidebar-footnote'>Total tools: {summary_info.get('total_tools', 0)} · Memory tools: {summary_info.get('memory_tools', 0)}</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.header("Upload Resume")
    uploaded_resume = st.file_uploader("Choose your resume", type=["pdf", "docx", "doc"])

    if uploaded_resume:
        result = upload_resume_to_backend(uploaded_resume)
        if result and result.get("status") == "success":
            st.success("Resume uploaded and processed.")
            st.session_state.resume_text = result.get("resume_text")
        else:
            st.error(f"Could not process resume: {result.get('message') if result else 'Unknown error'}")
            st.session_state.resume_text = None

    if st.session_state.resume_text:
        with st.expander("Resume snapshot", expanded=False):
            preview = st.session_state.resume_text[:500]
            if len(st.session_state.resume_text) > 500:
                preview += " ..."
            st.write(preview)

# Chat interface
user_input = st.chat_input("Ask about jobs...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.spinner("Processing..."):
        system_prompt = get_system_prompt_from_backend(st.session_state.resume_text)
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + st.session_state.chat_history
            reply = run_agent_on_backend(messages)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        else:
            st.error("Failed to construct system prompt. Cannot proceed.")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
