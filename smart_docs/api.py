from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from smart_docs.agent import SmartDocumentAgent
from smart_docs.rag import RAGService, SUPPORTED_EXTENSIONS
from smart_docs.schemas import ChatRequest, ChatResponse, DocumentInfo, HistoryItem

app = FastAPI(title="Smart Document Assistant", version="1.0.0")

rag_service = RAGService()
agent = SmartDocumentAgent(rag_service)


@app.on_event("startup")
def startup_ingest_samples() -> None:
    rag_service.ingest_existing_samples()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, int | str]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")

    final_path: Path | None = None
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        destination = Path(file.filename or f"upload{suffix}").name
        final_path = tmp_path.with_name(destination)
        tmp_path.replace(final_path)
        chunks = rag_service.ingest_upload(final_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if final_path and final_path.exists():
            final_path.unlink(missing_ok=True)

    return {"document_name": file.filename or "uploaded_document", "chunks_ingested": chunks}


@app.get("/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return rag_service.list_documents()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return agent.chat(
        session_id=request.session_id,
        message=request.message,
        top_k=request.top_k,
    )


@app.get("/chat/{session_id}/history", response_model=list[HistoryItem])
def chat_history(session_id: str) -> list[HistoryItem]:
    return agent.get_history(session_id)
