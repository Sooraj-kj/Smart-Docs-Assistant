from __future__ import annotations

import uuid
import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("SMART_DOCS_API_URL", "http://127.0.0.1:8000")


st.set_page_config(
    page_title="Smart Document Assistant",
    layout="wide",
)


def init_state() -> None:
    st.session_state.setdefault("api_base_url", DEFAULT_API_URL)
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
    if "selected_session_id" not in st.session_state:
        st.session_state.selected_session_id = st.session_state.session_id
    if "session_id_input" not in st.session_state:
        st.session_state.session_id_input = st.session_state.session_id
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "sessions" not in st.session_state:
        st.session_state.sessions = []
    if "sessions_loaded" not in st.session_state:
        st.session_state.sessions_loaded = False
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = 0


def api_url() -> str:
    return st.session_state.api_base_url.rstrip("/")


def request_json(method: str, path: str, **kwargs: Any) -> tuple[dict[str, Any] | list[Any] | None, str | None]:
    try:
        response = requests.request(method, f"{api_url()}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Could not connect to the FastAPI backend. Start it with: python main.py"
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        return None, f"API error: {detail}"
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {exc}"


def health_check() -> bool:
    data, error = request_json("GET", "/health")
    if error:
        st.error(error)
        return False
    return bool(data and data.get("status") == "ok")


def list_documents() -> list[dict[str, Any]]:
    data, error = request_json("GET", "/documents")
    if error:
        st.warning(error)
        return []
    return data if isinstance(data, list) else []


def list_chat_sessions(show_errors: bool = True) -> list[dict[str, Any]]:
    data, error = request_json("GET", "/chat/sessions")
    if error:
        if show_errors:
            st.warning(error)
        return []
    return data if isinstance(data, list) else []


def load_chat_history(session_id: str, show_errors: bool = True) -> None:
    data, error = request_json("GET", f"/chat/{session_id}/history")
    if error:
        if show_errors:
            st.warning(error)
        st.session_state.messages = []
        return
    history = data if isinstance(data, list) else []
    st.session_state.messages = [
        {"role": str(item.get("role", "assistant")), "content": str(item.get("content", ""))}
        for item in history
    ]
    st.session_state.last_trace = []
    st.session_state.last_sources = []


def refresh_chat_sessions(show_errors: bool = True) -> None:
    st.session_state.sessions = list_chat_sessions(show_errors=show_errors)
    st.session_state.sessions_loaded = True


def switch_session(session_id: str) -> None:
    st.session_state.session_id = session_id
    st.session_state.selected_session_id = session_id
    st.session_state.session_id_input = session_id
    load_chat_history(session_id)


def switch_selected_session() -> None:
    switch_session(st.session_state.selected_session_id)


def open_session_from_input() -> None:
    session_id = st.session_state.session_id_input.strip()
    if not session_id:
        return
    switch_session(session_id)


def upload_document(uploaded_file: Any) -> None:
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    data, error = request_json("POST", "/documents/upload", files=files)
    if error:
        st.error(error)
        return
    doc_name = data.get("document_name", uploaded_file.name)
    chunks = data.get("chunks_ingested", "?")
    st.success(f"Uploaded {doc_name} ({chunks} chunks ingested).")
    st.session_state.uploader_key += 1
    st.session_state.documents = list_documents()


def send_chat(message: str, top_k: int) -> None:
    payload = {
        "session_id": st.session_state.session_id,
        "message": message,
        "top_k": top_k,
    }
    data, error = request_json("POST", "/chat", json=payload)
    if error:
        st.session_state.messages.append({"role": "assistant", "content": error})
        return

    answer = str(data.get("answer", "No answer returned."))
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.last_sources = data.get("sources", [])
    st.session_state.last_trace = data.get("trace", [])
    refresh_chat_sessions(show_errors=False)


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No source citations returned for the latest answer.")
        return

    for source in sources:
        page = source.get("page")
        location = f"page {page}" if page else "text chunk"
        score = source.get("score")
        score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        st.markdown(
            f"**{source.get('document_name', 'unknown')}** · {location} · score `{score_text}`"
        )
        st.caption(f"chunk_id: {source.get('chunk_id', '')}")


def render_trace_value(value: Any) -> None:
    if isinstance(value, (dict, list)):
        st.json(value)
    elif value is None:
        st.caption("None")
    else:
        st.code(str(value), language="text")


def render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.info("No tools were called for the latest answer.")
        return

    for step in trace:
        with st.expander(f"Tool: {step.get('tool', 'unknown')}", expanded=False):
            st.markdown("**Input**")
            render_trace_value(step.get("input", {}))
            st.markdown("**Output**")
            render_trace_value(step.get("output"))


def new_session() -> None:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
    st.session_state.selected_session_id = st.session_state.session_id
    st.session_state.session_id_input = st.session_state.session_id
    st.session_state.messages = []
    st.session_state.last_trace = []
    st.session_state.last_sources = []


def refresh_documents() -> None:
    st.session_state.documents = list_documents()


init_state()

with st.sidebar:
    st.title("Smart Docs")
    st.caption("RAG assistant with tool traces")

    st.text_input(
        "FastAPI backend URL",
        key="api_base_url",
        help="Start the backend first with python main.py.",
    )

    if st.button("Check backend", width="stretch"):
        if health_check():
            st.success("Backend is online.")

    st.divider()

    if not st.session_state.sessions_loaded:
        refresh_chat_sessions(show_errors=False)

    st.subheader("Chats")
    session_labels = {
        session["session_id"]: (
            f"{session.get('title') or session['session_id']} "
            f"({session.get('message_count', 0)} messages)"
        )
        for session in st.session_state.sessions
    }
    session_options = list(session_labels)
    if st.session_state.session_id not in session_options:
        session_options.insert(0, st.session_state.session_id)

    st.selectbox(
        "Session history",
        session_options,
        key="selected_session_id",
        format_func=lambda session_id: session_labels.get(session_id, session_id),
        on_change=switch_selected_session,
    )
    st.text_input("Open or create session ID", key="session_id_input")
    st.button("Open session", width="stretch", on_click=open_session_from_input)
    st.button("New session", width="stretch", on_click=new_session)
    st.button("Refresh history", width="stretch", on_click=refresh_chat_sessions)

    st.divider()

    st.subheader("Upload")
    uploaded = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded and st.button("Upload document", width="stretch"):
        upload_document(uploaded)

    st.divider()

    st.subheader("Documents")
    st.button("Refresh documents", width="stretch", on_click=refresh_documents)

    docs = st.session_state.documents
    if docs:
        for doc in docs:
            st.markdown(f"**{doc.get('document_name')}**")
            st.caption(f"{doc.get('file_type')} · {doc.get('chunks')} chunks")
    else:
        st.caption("No documents found yet.")


st.title("Smart Document Assistant")
st.caption("Ask questions over uploaded documents. The latest answer shows citations and the tools the agent used.")

example_cols = st.columns(4)
examples = [
    "What are the parental leave rules?",
    "What is 15% of the operating budget?",
    "What MFA methods are approved?",
    "What is today's date and time?",
]

for col, example in zip(example_cols, examples):
    if col.button(example, width="stretch"):
        st.session_state.messages.append({"role": "user", "content": example})
        send_chat(example, top_k=5)
        st.rerun()

chat_col, detail_col = st.columns([2, 1])

with chat_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask a question about your documents...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                send_chat(prompt, top_k=5)
            st.markdown(st.session_state.messages[-1]["content"])
        st.rerun()

with detail_col:
    tabs = st.tabs(["Sources", "Trace"])
    with tabs[0]:
        render_sources(st.session_state.last_sources)
    with tabs[1]:
        render_trace(st.session_state.last_trace)
