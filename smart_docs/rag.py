from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3


class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class RAGService:
    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_STORE_DIR),
        )

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

        self.vector_store.add_texts(
            texts=[chunk.page_content for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
            ids=[str(chunk.metadata["chunk_id"]) for chunk in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        semantic_matches = self._semantic_search(query, top_k * 4)
        keyword_matches = self._keyword_search(query, top_k * 4)
        return self._merge_ranked_matches(semantic_matches, keyword_matches, top_k)

    def list_documents(self) -> list[DocumentInfo]:
        raw = self.vector_store.get(include=["metadatas"])
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

    def _load_and_chunk(self, path: Path) -> list[Document]:
        documents = self._load_documents(path)
        chunks = self.text_splitter.split_documents(documents)
        chunk_counts: dict[tuple[str, int], int] = {}

        for chunk in chunks:
            page = int(chunk.metadata.get("page", 0))
            key = (str(chunk.metadata["document_name"]), page)
            chunk_index = chunk_counts.get(key, 0)
            chunk_counts[key] = chunk_index + 1
            chunk.page_content = " ".join(chunk.page_content.split())
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_id"] = self._chunk_id(path, page, chunk_index, chunk.page_content)

        return [chunk for chunk in chunks if chunk.page_content]

    @staticmethod
    def _load_documents(path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents = PyMuPDFLoader(str(path)).load()
        elif suffix in {".txt", ".md"}:
            documents = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        for document in documents:
            page = int(document.metadata.get("page", 0)) + 1 if suffix == ".pdf" else 0
            document.metadata = {
                "document_name": path.name,
                "source_path": str(path),
                "file_type": suffix.lstrip("."),
                "page": page,
            }
        return documents

    def _semantic_search(self, query: str, top_k: int) -> dict[str, dict[str, Any]]:
        matches: dict[str, dict[str, Any]] = {}
        for document, distance in self.vector_store.similarity_search_with_score(query, k=top_k):
            chunk_id = str(document.metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            matches[chunk_id] = {
                "text": document.page_content,
                "metadata": document.metadata,
                "semantic_score": 1.0 / (1.0 + float(distance)),
            }
        return matches

    def _keyword_search(self, query: str, top_k: int) -> dict[str, dict[str, Any]]:
        raw = self.vector_store.get(include=["documents", "metadatas"])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])
        if not documents or not metadatas:
            return {}

        query_terms = self._tokenize(query)
        if not query_terms:
            return {}

        tokenized_documents = [self._tokenize(document or "") for document in documents]
        scores = self._bm25_scores(query_terms, tokenized_documents)
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

        matches: dict[str, dict[str, Any]] = {}
        for index in ranked_indexes:
            if scores[index] <= 0:
                continue
            metadata = metadatas[index] or {}
            chunk_id = str(metadata.get("chunk_id", ""))
            if not chunk_id:
                continue
            matches[chunk_id] = {
                "text": documents[index],
                "metadata": metadata,
                "keyword_score": scores[index],
            }
        return matches

    @staticmethod
    def _merge_ranked_matches(
        semantic_matches: dict[str, dict[str, Any]],
        keyword_matches: dict[str, dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        max_keyword_score = max(
            (match.get("keyword_score", 0.0) for match in keyword_matches.values()),
            default=0.0,
        )
        all_chunk_ids = set(semantic_matches) | set(keyword_matches)
        merged = []

        for chunk_id in all_chunk_ids:
            semantic_match = semantic_matches.get(chunk_id, {})
            keyword_match = keyword_matches.get(chunk_id, {})
            keyword_score = float(keyword_match.get("keyword_score", 0.0))
            normalized_keyword_score = keyword_score / max_keyword_score if max_keyword_score else 0.0
            combined_score = (
                SEMANTIC_WEIGHT * float(semantic_match.get("semantic_score", 0.0))
                + KEYWORD_WEIGHT * normalized_keyword_score
            )
            best_match = semantic_match or keyword_match
            merged.append(
                {
                    "text": best_match["text"],
                    "metadata": best_match["metadata"],
                    "score": combined_score,
                    "semantic_score": semantic_match.get("semantic_score", 0.0),
                    "keyword_score": keyword_score,
                }
            )

        return sorted(merged, key=lambda match: match["score"], reverse=True)[:top_k]

    @staticmethod
    def _bm25_scores(
        query_terms: list[str],
        tokenized_documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[float]:
        document_count = len(tokenized_documents)
        if document_count == 0:
            return []

        average_length = sum(len(document) for document in tokenized_documents) / document_count
        document_frequencies = Counter(
            term
            for document in tokenized_documents
            for term in set(document)
        )

        scores = []
        for document in tokenized_documents:
            term_frequencies = Counter(document)
            document_length = len(document) or 1
            score = 0.0
            for term in query_terms:
                if term not in term_frequencies:
                    continue
                idf = math.log(1 + (document_count - document_frequencies[term] + 0.5) / (document_frequencies[term] + 0.5))
                frequency = term_frequencies[term]
                denominator = frequency + k1 * (1 - b + b * document_length / (average_length or 1))
                score += idf * (frequency * (k1 + 1)) / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 1]

    @staticmethod
    def _chunk_id(path: Path, page: int, chunk_index: int, text: str) -> str:
        digest = hashlib.sha1(f"{path.name}:{page}:{chunk_index}:{text}".encode("utf-8")).hexdigest()
        return digest
