import json
import os
import html
from datetime import datetime
from typing import Any, Dict, List

import requests
import streamlit as st
from sseclient import SSEClient

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:5001")
PAGE_TITLE = "Job Search Copilot"

PERSONA_OPTIONS: Dict[str, Dict[str, Any]] = {
    "career_strategist": {
        "label": "Career Strategist",
        "description": "Long-range planning, bold pivots, and leadership pathways.",
        "actions": [
            {
                "label": "Map my next promotion",
                "prompt": "Help me outline a 12-month plan to reach the next level in my career, including projects and skills to prioritize.",
            },
            {
                "label": "Explore adjacent roles",
                "prompt": "Suggest 3 adjacent job titles I can pivot into given my current background, and what gaps I'd need to close.",
            },
            {
                "label": "Assess industry trends",
                "prompt": "Which industries are growing fastest for my skillset, and how should I reposition to ride that wave?",
            },
        ],
    },
    "application_optimizer": {
        "label": "Application Optimizer",
        "description": "High-efficiency search, ATS alignment, and volume tactics.",
        "actions": [
            {
                "label": "Batch apply plan",
                "prompt": "Design a weekly application plan that maximizes relevant postings and keeps me consistent with follow-through.",
            },
            {
                "label": "Optimize resume keywords",
                "prompt": "Analyze my resume for missing keywords based on typical job postings for my target role.",
            },
            {
                "label": "Track job leads",
                "prompt": "Create a job lead tracker structure that helps me measure response rates and follow-ups.",
            },
        ],
    },
    "interview_coach": {
        "label": "Interview Coach",
        "description": "Story refinement, behavioral drills, and mock interviewer feedback.",
        "actions": [
            {
                "label": "Mock interview drill",
                "prompt": "Run a mock behavioral interview with me focused on leadership questions and provide feedback after each answer.",
            },
            {
                "label": "Story bank setup",
                "prompt": "Help me build a STAR story bank covering impact, conflict, and problem-solving scenarios from my experience.",
            },
            {
                "label": "Anticipate objections",
                "prompt": "Identify likely interviewer concerns about my background and coach me on addressing them confidently.",
            },
        ],
    },
}

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
THEME_CSS = """
<style>
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}
[data-testid="stChatMessageUser"] > div {
    background-color: #1d4ed8;
    color: #ffffff;
}
[data-testid="stChatMessageAssistant"] > div {
    background-color: #ffffff;
    color: #1f2937;
    border: 1px solid #e2e8f0;
}
.sidebar .sidebar-content, [data-testid="stSidebar"] {
    background-color: #ffffff;
}
.hero-card {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: #ffffff;
    padding: 1.6rem 2rem;
    border-radius: 20px;
    box-shadow: 0 24px 60px rgba(37, 99, 235, 0.35);
    margin-bottom: 1.4rem;
}
.hero-card h1 {
    margin: 0;
    font-size: 1.8rem;
    font-weight: 600;
}
.hero-card p {
    margin-top: 0.5rem;
    margin-bottom: 0;
    font-size: 1rem;
    opacity: 0.9;
}
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin-bottom: 1.2rem;
}
.metric-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 16px 36px rgba(15, 23, 42, 0.08);
}
.metric-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #0f172a;
}
.metric-caption {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.3rem;
}
.timeline-card {
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    padding: 1.25rem 1.4rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.07);
}
.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
}
.timeline-title {
    font-weight: 600;
    color: #1f2937;
    font-size: 1rem;
}
.timeline-items {
    display: grid;
    gap: 0.75rem;
}
.timeline-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    padding: 0.6rem 0.75rem;
    border-radius: 12px;
}
.timeline-item.tool_start {
    background: rgba(37, 99, 235, 0.08);
    color: #1d4ed8;
}
.timeline-item.tool_end {
    background: rgba(34, 197, 94, 0.12);
    color: #047857;
}
.timeline-time {
    font-size: 0.75rem;
    color: #64748b;
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# --- Session State Initialization ---
if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "initialization" not in st.session_state:
    st.session_state.initialization = {"status": "pending", "error": None, "metadata": None}

if "persona" not in st.session_state:
    st.session_state.persona = "career_strategist"

if "last_assistant_message" not in st.session_state:
    st.session_state.last_assistant_message = ""

if "quick_action_trigger" not in st.session_state:
    st.session_state.quick_action_trigger = None

# --- Helpers ---
def initialize_backend():
    """Ping backend for initialization status."""
    try:
        response = requests.post(f"{BACKEND_URL}/initialize", timeout=30)
        response.raise_for_status()
        st.session_state.initialization = {
            "status": "ready",
            "error": None,
            "metadata": response.json(),
        }
    except requests.RequestException as exc:
        st.session_state.initialization = {
            "status": "error",
            "error": str(exc),
            "metadata": None,
        }

def reset_conversation():
    st.session_state.conversation = []
    st.session_state.display_messages = []
    st.session_state.last_assistant_message = ""


def format_timestamp(timestamp: str | None) -> str:
    if not timestamp:
        return ""
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        return ""
    return dt.strftime("%H:%M:%S")


def record_display_message(role: str, content: str, message_type: str | None = None) -> None:
    st.session_state.display_messages.append(
        {
            "role": role,
            "content": content,
            "type": message_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    if role == "assistant" and message_type is None:
        st.session_state.last_assistant_message = content


def render_hero_banner() -> None:
    persona_info = PERSONA_OPTIONS.get(st.session_state.persona, {})
    persona_label = persona_info.get("label", "Job Search Copilot")
    persona_subtitle = persona_info.get(
        "description",
        "Your AI partner for uncovering roles, tailoring applications, and tracking hiring momentum.",
    )
    st.markdown(
        """
        <div class="hero-card">
            <h1>Job Search Copilot</h1>
            <p>{subtitle}</p>
        </div>
        """.format(subtitle=html.escape(persona_label + " · " + persona_subtitle)),
        unsafe_allow_html=True,
    )


def render_session_metrics() -> None:
    total_messages = len(st.session_state.conversation)
    user_messages = sum(1 for msg in st.session_state.conversation if msg["role"] == "user")
    assistant_messages = total_messages - user_messages
    resume_status = "Uploaded" if st.session_state.resume_text else "Not uploaded"
    tools_used = sum(1 for msg in st.session_state.display_messages if msg.get("type") == "tool_start")

    metrics_html = f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-title">Conversation Turns</div>
            <div class="metric-value">{total_messages}</div>
            <div class="metric-caption">{user_messages} from you · {assistant_messages} from Copilot</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Resume Context</div>
            <div class="metric-value">{html.escape(resume_status)}</div>
            <div class="metric-caption">Upload a resume to unlock tailored guidance</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Tools Triggered</div>
            <div class="metric-value">{tools_used}</div>
            <div class="metric-caption">Tracks MCP tool usage this session</div>
        </div>
    </div>
    """
    st.markdown(metrics_html, unsafe_allow_html=True)


def render_resume_health() -> None:
    st.markdown("#### Resume Signal Strength")
    if not st.session_state.resume_text:
        st.info("Upload a resume to analyze keyword density, length, and freshness.")
        return

    words = [w for w in st.session_state.resume_text.split() if w.isalpha()]
    word_count = len(words)
    unique_word_count = len({w.lower() for w in words})
    freshness_score = min(20, int(unique_word_count / 10))
    length_score = min(50, int(word_count / 12))
    balance_penalty = max(0, 30 - abs(word_count - 450) // 5)
    total_score = max(30, min(95, length_score + freshness_score + balance_penalty))

    st.progress(total_score / 100)
    st.caption(
        f"Word count: {word_count} · Unique keywords: {unique_word_count} · Health score: {total_score}/100"
    )


def render_quick_actions() -> None:
    persona = st.session_state.persona
    actions = PERSONA_OPTIONS.get(persona, {}).get("actions", [])
    if not actions:
        return

    st.markdown("##### Quick Actions")
    columns = st.columns(len(actions))
    for idx, action in enumerate(actions):
        col = columns[idx]
        if col.button(action["label"], use_container_width=True, key=f"qa_{persona}_{idx}"):
            st.session_state.quick_action_trigger = action["prompt"]


def extract_job_cards(text: str) -> List[Dict[str, str]]:
    jobs: List[Dict[str, str]] = []
    current: Dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-*•")
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if key in {"role", "title", "position"}:
                if current:
                    jobs.append(current)
                    current = {}
                current["title"] = value
            elif key == "company":
                current["company"] = value
            elif key == "location":
                current["location"] = value
            elif key in {"match", "why it matches", "fit"}:
                current["match"] = value
            elif key in {"salary", "compensation"}:
                current["salary"] = value
        elif any(prefix in line.lower() for prefix in ["role", "title"]):
            if current:
                jobs.append(current)
            current = {"title": line}
    if current:
        jobs.append(current)
    return jobs[:3]


def render_job_spotlight() -> None:
    message = st.session_state.last_assistant_message
    if not message:
        return

    job_cards = extract_job_cards(message)
    if not job_cards:
        return

    st.markdown("#### Job Match Spotlight")
    for job in job_cards:
        title = html.escape(job.get("title", "Role TBD"))
        company = html.escape(job.get("company", "Company TBD"))
        location = html.escape(job.get("location", "Location flexible"))
        match = html.escape(job.get("match", "Relevant fit details pending."))
        salary = html.escape(job.get("salary", ""))
        salary_line = f"<div><strong>Compensation:</strong> {salary}</div>" if salary else ""
        st.markdown(
            f"""
            <div class="timeline-card">
                <div class="timeline-header">
                    <span class="timeline-title">{title}</span>
                    <span class="timeline-time">{company}</span>
                </div>
                <div>{location}</div>
                <div style="margin-top:0.5rem;">{match}</div>
                {salary_line}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_tool_timeline() -> None:
    tool_events = [msg for msg in st.session_state.display_messages if msg.get("type") in {"tool_start", "tool_end"}]
    st.markdown("#### Tool Activity")

    if not tool_events:
        st.info("No tool runs yet. Actions will appear here once the copilot invokes a tool.")
        return

    recent_events = list(reversed(tool_events[-6:]))
    items_html = []
    for event in recent_events:
        name = html.escape(event.get("content", "Unknown"))
        event_type = event.get("type", "tool_start")
        timestamp = format_timestamp(event.get("timestamp"))
        items_html.append(
            f"<div class='timeline-item {event_type}'>"
            f"<span>{name}</span>"
            f"<span class='timeline-time'>{timestamp}</span>"
            f"</div>"
        )

    timeline_html = """
    <div class="timeline-card">
        <div class="timeline-header">
            <span class="timeline-title">Recent Automations</span>
        </div>
        <div class="timeline-items">
    """ + "".join(items_html) + """
        </div>
    </div>
    """
    st.markdown(timeline_html, unsafe_allow_html=True)


def render_message(role: str, content: str, message_type: str | None = None):
    """Render a chat or tool message."""
    avatar = "👤" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        if message_type == "tool_start":
            st.markdown(f"🔧 Using tool: **{content}**")
        elif message_type == "tool_end":
            st.markdown(f"✅ Finished tool: **{content}**")
        else:
            st.markdown(content)


def upload_resume(file):
    """Upload resume to backend and store extracted text."""
    files = {"file": (file.name, file.getvalue(), file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/upload_resume", files=files, timeout=60)
        response.raise_for_status()
        payload = response.json()
        st.session_state.resume_text = payload.get("resume_text")
        st.success("Resume uploaded and processed successfully.")
    except requests.RequestException as exc:
        st.error(f"Resume upload failed: {exc}")


def stream_chat(prompt: str) -> None:
    """Send chat request to backend and display streaming response."""
    payload = {
        "messages": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.conversation
        ] + [{"role": "user", "content": prompt}],
        "resume_text": st.session_state.resume_text,
        "persona": st.session_state.persona,
    }

    st.session_state.conversation.append({"role": "user", "content": prompt})
    record_display_message("user", prompt)
    render_message("user", prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        tool_feedback = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{BACKEND_URL}/chat/stream",
                json=payload,
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                client = SSEClient(response)

                for event in client.events():
                    if event.event == "message":
                        try:
                            data = json.loads(event.data)
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type")
                        if event_type == "token":
                            token = data.get("content", "")
                            full_response += token
                            message_placeholder.markdown(full_response + "▌")
                        elif event_type == "final":
                            final_text = data.get("content", "")
                            if final_text and not full_response:
                                full_response = final_text
                            break
                        elif event_type == "error":
                            st.error(data.get("content", "An unknown error occurred."))
                            break
                    elif event.event == "tool_start":
                        try:
                            data = json.loads(event.data)
                        except json.JSONDecodeError:
                            continue

                        tool_name = data.get("name", "unknown tool")
                        tool_feedback.info(f"🔧 Using tool: **{tool_name}**")
                        record_display_message("assistant", tool_name, "tool_start")
                    elif event.event == "tool_end":
                        try:
                            data = json.loads(event.data)
                        except json.JSONDecodeError:
                            continue

                        tool_name = data.get("name", "unknown tool")
                        tool_feedback.success(f"✅ Finished tool: **{tool_name}**")
                        record_display_message("assistant", tool_name, "tool_end")
                    elif event.event == "error":
                        try:
                            data = json.loads(event.data)
                            st.error(data.get("content", "An unknown error occurred."))
                        except json.JSONDecodeError:
                            st.error("An unknown error occurred.")
                        break
                    elif event.event == "end":
                        break

        except requests.RequestException as exc:
            st.error(f"Connection to backend failed: {exc}")

        message_placeholder.markdown(full_response)
        st.session_state.conversation.append({"role": "assistant", "content": full_response})
        record_display_message("assistant", full_response)


# --- Main Layout ---
with st.sidebar:
    st.markdown("### Persona")
    persona_keys = list(PERSONA_OPTIONS.keys())
    current_index = persona_keys.index(st.session_state.persona)
    selected_key = st.selectbox(
        "Choose copilot style",
        options=persona_keys,
        index=current_index,
        format_func=lambda key: PERSONA_OPTIONS[key]["label"],
    )
    st.session_state.persona = selected_key
    st.caption(PERSONA_OPTIONS[selected_key]["description"])

    st.divider()

    st.header("Session Controls")
    if st.button("New Session", use_container_width=True):
        reset_conversation()
        st.toast("Started a fresh session.")

    st.markdown("### Resume")
    uploaded_file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"], accept_multiple_files=False)
    if uploaded_file is not None:
        upload_resume(uploaded_file)

    st.markdown("### Backend Status")
    if st.session_state.initialization["status"] == "pending":
        with st.spinner("Initializing copilot..."):
            initialize_backend()
    status = st.session_state.initialization.get("status")
    if status == "ready":
        st.success("Backend ready")
    elif status == "error":
        st.error(f"Backend error: {st.session_state.initialization['error']}")
    else:
        st.info("Waiting for backend")

render_hero_banner()
st.caption("A modern job search partner with intelligent tooling.")

render_quick_actions()

overview_col, timeline_col = st.columns([2, 1])
with overview_col:
    render_session_metrics()
    render_resume_health()
    render_job_spotlight()
with timeline_col:
    render_tool_timeline()

for message in st.session_state.display_messages:
    msg_type = message.get("type")
    render_message(message["role"], message["content"], msg_type)

if st.session_state.quick_action_trigger:
    prompt = st.session_state.quick_action_trigger
    st.session_state.quick_action_trigger = None
    stream_chat(prompt)
elif prompt := st.chat_input("How can I assist you today?"):
    stream_chat(prompt)