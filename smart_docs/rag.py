from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

from smart_docs.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    UPLOAD_DIR,
    VECTOR_STORE_DIR,
)
from smart_docs.schemas import DocumentInfo


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class TextChunk:
    text: str
    metadata: dict[str, Any]


class EmbeddingManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()


class RAGService:
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)

    def ingest_existing_samples(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for folder in ("texts", "pdf"):
            source_dir = UPLOAD_DIR.parent / folder
            if source_dir.exists():
                for path in source_dir.iterdir():
                    if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        counts[path.name] = self.ingest_file(path, copy_to_uploads=False)
        return counts

    def ingest_upload(self, source_path: Path) -> int:
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Only PDF, TXT, and MD files are supported.")

        destination = UPLOAD_DIR / source_path.name
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)
        return self.ingest_file(destination, copy_to_uploads=False)

    def ingest_file(self, path: Path, copy_to_uploads: bool = True) -> int:
        if copy_to_uploads:
            return self.ingest_upload(path)

        chunks = self._load_and_chunk(path)
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        query_embedding = self.embedding_manager.embed_texts([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        matches = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "score": float(distance),
                }
            )
        return matches

    def list_documents(self) -> list[DocumentInfo]:
        raw = self.collection.get(include=["metadatas"])
        grouped: dict[str, dict[str, Any]] = {}

        for metadata in raw.get("metadatas", []):
            if not metadata:
                continue
            name = str(metadata.get("document_name", "unknown"))
            grouped.setdefault(
                name,
                {
                    "document_name": name,
                    "source_path": str(metadata.get("source_path", "")),
                    "file_type": str(metadata.get("file_type", "")),
                    "chunks": 0,
                },
            )
            grouped[name]["chunks"] += 1

        return [DocumentInfo(**item) for item in sorted(grouped.values(), key=lambda x: x["document_name"])]

    def _load_and_chunk(self, path: Path) -> list[TextChunk]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = self._read_pdf_pages(path)
        elif suffix in {".txt", ".md"}:
            pages = [(None, path.read_text(encoding="utf-8", errors="replace"))]
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        chunks: list[TextChunk] = []
        for page_number, text in pages:
            for chunk_index, chunk_text in enumerate(self._split_text(text)):
                chunk_id = self._chunk_id(path, page_number, chunk_index, chunk_text)
                page_value = page_number if page_number is not None else 0
                chunks.append(
                    TextChunk(
                        text=chunk_text,
                        metadata={
                            "chunk_id": chunk_id,
                            "document_name": path.name,
                            "source_path": str(path),
                            "file_type": suffix.lstrip("."),
                            "page": page_value,
                            "chunk_index": chunk_index,
                        },
                    )
                )
        return chunks

    @staticmethod
    def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
        pages: list[tuple[int, str]] = []
        with fitz.open(path) as pdf:
            for index, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()
                if text:
                    pages.append((index, text))
        return pages

    @staticmethod
    def _split_text(text: str) -> list[str]:
        normalized = " ".join(text.split())
        if not normalized:
            return []

        chunks = []
        start = 0
        while start < len(normalized):
            end = min(start + CHUNK_SIZE, len(normalized))
            if end < len(normalized):
                boundary = normalized.rfind(" ", start, end)
                if boundary > start + CHUNK_SIZE // 2:
                    end = boundary
            chunks.append(normalized[start:end].strip())
            if end >= len(normalized):
                break
            start = max(end - CHUNK_OVERLAP, 0)
        return chunks

    @staticmethod
    def _chunk_id(path: Path, page: int | None, chunk_index: int, text: str) -> str:
        digest = hashlib.sha1(f"{path.name}:{page}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
        return digest
