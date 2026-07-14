from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from smart_docs.chat_store import SQLiteChatStore
from smart_docs.config import DEFAULT_LLM_MODEL, DEFAULT_TOP_K
from smart_docs.rag import RAGService
from smart_docs.schemas import ChatResponse, ChatSession, HistoryItem, SourceCitation
from smart_docs.tools import calculator, current_datetime, search_documents


class SmartDocumentAgent:
    def __init__(self, rag: RAGService):
        self.rag = rag
        self.store = SQLiteChatStore()
        self.llm = self._build_llm()

    def chat(self, session_id: str, message: str, top_k: int = DEFAULT_TOP_K) -> ChatResponse:
        self.store.ensure_session(session_id)
        history = self.store.get_history(session_id)
        trace: list[dict[str, Any]] = []
        retrieved: list[dict[str, Any]] = []
        calculation_result: float | None = None
        now: str | None = None

        retrieval_query = self._make_retrieval_query(history, message)
        if self._should_search(message):
            retrieved = search_documents(self.rag, retrieval_query, top_k)
            if not self._has_query_evidence(message, retrieved):
                retrieved = []
            trace.append(
                {
                    "tool": "search_documents",
                    "input": {"query": retrieval_query, "top_k": top_k},
                    "output": self._trace_sources(retrieved),
                }
            )

        expression = self._extract_math_expression(message, retrieved)
        if expression:
            calculation_result = calculator(expression)
            trace.append(
                {
                    "tool": "calculator",
                    "input": {"expression": expression},
                    "output": calculation_result,
                }
            )

        if self._asks_for_time(message):
            now = current_datetime()
            trace.append(
                {
                    "tool": "current_datetime",
                    "input": {"timezone": "Asia/Kolkata"},
                    "output": now,
                }
            )

        answer = self._generate_answer(
            message=message,
            history=history,
            retrieved=retrieved,
            calculation_result=calculation_result,
            current_time=now,
        )

        self.store.append_messages(
            session_id,
            [
                HistoryItem(role="user", content=message),
                HistoryItem(role="assistant", content=answer),
            ],
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            sources=self._citations(retrieved),
            trace=trace,
        )

    def get_history(self, session_id: str) -> list[HistoryItem]:
        return self.store.get_history(session_id)

    def list_sessions(self) -> list[ChatSession]:
        return self.store.list_sessions()

    @staticmethod
    def _build_llm() -> ChatGroq | None:
        if not os.getenv("GROQ_API_KEY"):
            return None
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", DEFAULT_LLM_MODEL),
            temperature=0.1,
            max_tokens=700,
            timeout=30,
            max_retries=2,
        )

    @staticmethod
    def _should_search(message: str) -> bool:
        keywords = [
            "document",
            "policy",
            "report",
            "manual",
            "leave",
            "budget",
            "revenue",
            "password",
            "mfa",
            "security",
            "what",
            "how",
            "summarize",
            "second one",
        ]
        lowered = message.lower()
        return any(keyword in lowered for keyword in keywords)

    @staticmethod
    def _asks_for_time(message: str) -> bool:
        lowered = message.lower()
        return "date" in lowered or "time" in lowered or "today" in lowered

    @staticmethod
    def _make_retrieval_query(history: list[HistoryItem], message: str) -> str:
        recent = " ".join(item.content for item in history[-4:])
        return f"{recent} {message}".strip()

    @staticmethod
    def _extract_math_expression(message: str, retrieved: list[dict[str, Any]]) -> str | None:
        lowered = message.lower()
        context = " ".join(item["text"] for item in retrieved)

        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(?:the\s+)?([a-zA-Z\s]+)", lowered)
        if percent_match:
            percent = percent_match.group(1)
            target_phrase = percent_match.group(2).strip()
            amount = SmartDocumentAgent._find_amount_near_phrase(context, target_phrase)
            if amount is not None:
                return f"{percent}/100*{amount}"

        explicit_math = re.findall(r"[0-9][0-9,\.\s\+\-\*\/\(\)%$]*", message)
        candidate = "".join(explicit_math).strip()
        if any(op in candidate for op in ["+", "-", "*", "/", "%"]) and candidate:
            return candidate
        return None

    @staticmethod
    def _has_query_evidence(message: str, retrieved: list[dict[str, Any]]) -> bool:
        if not retrieved:
            return False

        stopwords = {
            "what",
            "the",
            "and",
            "for",
            "with",
            "from",
            "does",
            "this",
            "that",
            "have",
            "into",
            "when",
            "where",
            "which",
            "about",
            "company",
            "document",
            "documents",
            "policy",
            "report",
            "manual",
            "mentioned",
            "approved",
            "methods",
        }
        terms = [
            term
            for term in re.findall(r"[a-zA-Z0-9]+", message.lower())
            if len(term) > 2 and term not in stopwords
        ]
        if not terms:
            return True

        context = " ".join(item["text"].lower() for item in retrieved[:3])
        return any(term in context for term in terms)

    @staticmethod
    def _find_amount_near_phrase(context: str, phrase: str) -> float | None:
        compact_context = " ".join(context.split())
        phrase_words = [word for word in phrase.split() if len(word) > 2]
        phrase_pattern = r"\b" + r"\s+".join(re.escape(word) for word in phrase_words) + r"\b"
        money_pattern = r"(\$\d[\d,]*(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)"
        direct_match = re.search(
            phrase_pattern + r".{0,120}?" + money_pattern,
            compact_context,
            flags=re.IGNORECASE,
        )
        if direct_match:
            return float(direct_match.group(1).replace("$", "").replace(",", ""))

        money_matches = list(re.finditer(money_pattern, compact_context))
        if not money_matches:
            return None

        best_match = None
        best_score = -1
        for match in money_matches:
            window = compact_context[max(0, match.start() - 120) : match.end() + 120].lower()
            score = sum(1 for word in phrase_words if word in window)
            if score > best_score:
                best_score = score
                best_match = match.group(0)

        if best_match is None:
            return None
        return float(best_match.replace("$", "").replace(",", ""))

    def _generate_answer(
        self,
        message: str,
        history: list[HistoryItem],
        retrieved: list[dict[str, Any]],
        calculation_result: float | None,
        current_time: str | None,
    ) -> str:
        context = self._format_context(retrieved)
        extras = []
        if calculation_result is not None:
            extras.append(f"Calculator result: {calculation_result:g}")
        if current_time is not None:
            extras.append(f"Current datetime: {current_time}")

        if self.llm is None:
            return self._fallback_answer(retrieved, calculation_result, current_time)

        history_text = "\n".join(f"{item.role}: {item.content}" for item in history[-6:])
        system = SystemMessage(
            content=(
                "You are a Smart Document Assistant. Answer using only the retrieved context and tool outputs. "
                "Cite sources as [document_name, page/chunk]. If the context does not contain the answer, "
                "say you don't know based on the provided documents. Be concise."
            )
        )
        human = HumanMessage(
            content=(
                f"Conversation history:\n{history_text or 'No prior messages.'}\n\n"
                f"Retrieved context:\n{context or 'No retrieved context.'}\n\n"
                f"Tool outputs:\n{chr(10).join(extras) or 'No extra tool outputs.'}\n\n"
                f"User question: {message}"
            )
        )
        try:
            return self.llm.invoke([system, human]).content
        except Exception:
            return self._fallback_answer(retrieved, calculation_result, current_time)

    @staticmethod
    def _fallback_answer(
        retrieved: list[dict[str, Any]],
        calculation_result: float | None,
        current_time: str | None,
    ) -> str:
        parts = []
        if retrieved:
            top = retrieved[0]
            metadata = top["metadata"]
            location = f"{metadata.get('document_name')} chunk {metadata.get('chunk_index')}"
            if metadata.get("page"):
                location = f"{metadata.get('document_name')} page {metadata.get('page')}"
            parts.append(f"Most relevant context from {location}: {top['text']}")
        else:
            parts.append("I don't know based on the provided documents.")
        if calculation_result is not None:
            parts.append(f"Calculator result: {calculation_result:g}.")
        if current_time is not None:
            parts.append(f"Current datetime: {current_time}.")
        return "\n\n".join(parts)

    @staticmethod
    def _format_context(retrieved: list[dict[str, Any]]) -> str:
        blocks = []
        for item in retrieved:
            metadata = item["metadata"]
            page = metadata.get("page")
            page_label = f"page {page}" if page else f"chunk {metadata.get('chunk_index')}"
            blocks.append(f"[{metadata.get('document_name')}, {page_label}]\n{item['text']}")
        return "\n\n".join(blocks)

    @staticmethod
    def _citations(retrieved: list[dict[str, Any]]) -> list[SourceCitation]:
        citations = []
        seen = set()
        for item in retrieved:
            metadata = item["metadata"]
            key = metadata.get("chunk_id")
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                SourceCitation(
                    document_name=str(metadata.get("document_name", "unknown")),
                    chunk_id=str(metadata.get("chunk_id", "")),
                    page=metadata.get("page") or None,
                    score=item.get("score"),
                )
            )
        return citations

    @staticmethod
    def _trace_sources(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "document_name": item["metadata"].get("document_name"),
                "page": item["metadata"].get("page"),
                "chunk_index": item["metadata"].get("chunk_index"),
                "score": item.get("score"),
            }
            for item in retrieved
        ]
