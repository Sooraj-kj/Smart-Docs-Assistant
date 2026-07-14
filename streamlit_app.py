from __future__ import annotations

import uuid
from typing import Any

import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Smart Document Assistant",
    layout="wide",
)


def init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
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


def render_trace(trace: list[dict[str, Any]]) -> None:
    if not trace:
        st.info("No tools were called for the latest answer.")
        return

    for step in trace:
        with st.expander(f"Tool: {step.get('tool', 'unknown')}", expanded=False):
            st.markdown("**Input**")
            st.json(step.get("input", {}))
            st.markdown("**Output**")
            st.json(step.get("output", {}))


def new_session() -> None:
    st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
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
        value=DEFAULT_API_URL,
        key="api_base_url",
        help="Start the backend first with python main.py.",
    )

    if st.button("Check backend", use_container_width=True):
        if health_check():
            st.success("Backend is online.")

    st.divider()

    st.text_input("Session ID", key="session_id")
    st.button("New session", use_container_width=True, on_click=new_session)

    st.divider()

    st.subheader("Upload")
    uploaded = st.file_uploader(
        "PDF, TXT, or MD",
        type=["pdf", "txt", "md"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    if uploaded and st.button("Upload document", use_container_width=True):
        upload_document(uploaded)

    st.divider()

    st.subheader("Documents")
    st.button("Refresh documents", use_container_width=True, on_click=refresh_documents)

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
    if col.button(example, use_container_width=True):
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