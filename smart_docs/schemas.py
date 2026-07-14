from typing import Any

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    document_name: str
    chunk_id: str
    page: int | None = None
    score: float | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=10)


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceCitation] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)


class ChatSession(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class DocumentInfo(BaseModel):
    document_name: str
    source_path: str
    file_type: str
    chunks: int


class HistoryItem(BaseModel):
    role: str
    content: str
