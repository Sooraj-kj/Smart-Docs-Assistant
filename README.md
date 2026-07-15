# Smart Document Assistant

A RAG-powered AI assistant that ingests documents, answers grounded questions with citations, and uses tools for document search, calculations, and date/time queries.

## Features

- Upload and ingest PDF, TXT, and Markdown files.
- Chunk documents with overlapping character windows.
- Generate local sentence-transformer embeddings.
- Store embeddings with Chroma and retrieve with a hybrid semantic + BM25 keyword strategy.
- Answer questions using retrieved context and source citations.
- Agent tool trace showing which tools were called.
- Calculator support for multi-step questions such as "What is 15% of the operating budget?"
- Persistent conversation memory by `session_id` with SQLite-backed chat history.
- FastAPI REST API.
- Streamlit frontend.
- Small keyword-based evaluation script.

## Architecture

```text
Upload/API
   |
   v
LangChain loaders -> RecursiveCharacterTextSplitter -> SentenceTransformer embeddings -> Chroma vector DB
                                                              |
                                                              v
User chat -> LangChain agent -> tools: search_documents, calculator, current_datetime
                                                              |
                                                              v
                                                   Groq LLM grounded answer
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq key to `.env`:

```text
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
```

If `GROQ_API_KEY` is not set, the API still runs with a simple extractive fallback answer, but LLM-quality responses need Groq.

## Run

### Docker Compose

For one-command startup, copy `.env.example` to `.env`, add your Groq key if you have one, then run:

```bash
docker compose up --build
```

Open the UI:

```text
http://localhost:8501
```

The API is available at:

```text
http://localhost:8000/docs
```

Compose starts two services: `api` for FastAPI and `frontend` for Streamlit. The local `data/` folder is mounted into the containers so uploaded documents, Chroma indexes, and SQLite chat history persist across restarts.

### Local Python

Start the FastAPI backend:

```bash
python main.py
```

Open the API docs:

```text
http://localhost:8000/docs
```

The app ingests the included sample documents from `data/texts` and `data/pdf` on startup.

Start the Streamlit frontend in a second terminal:

```bash
streamlit run streamlit_app.py
```

Open the UI:

```text
http://localhost:8501
```

## API

### Upload a Document

```bash
curl -X POST "http://localhost:8000/documents/upload" ^
  -F "file=@data/texts/leave_policy.txt"
```

### List Documents

```bash
curl "http://localhost:8000/documents"
```

### Chat

```bash
curl -X POST "http://localhost:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo\",\"message\":\"What are the parental leave rules?\"}"
```

### Chat History

```bash
curl "http://localhost:8000/chat/demo/history"
```

## Example Queries

- `What are the parental leave rules?`
- `What is the Q4 2025 operating budget?`
- `What is 15% of the operating budget mentioned in the report?`
- `What MFA methods are approved?`
- `What is the company pet policy?`

The last query should return an "I don't know based on the provided documents" style answer.

## Evaluation

```bash
python scripts/eval_rag.py
```

This runs five sample questions and checks for expected keywords. It is intentionally lightweight so it can run without paid evaluation tooling.

## Design Decisions

**Chunking strategy:** Documents are loaded with LangChain loaders and split with `RecursiveCharacterTextSplitter` into 1000-character chunks with 200-character overlap. This keeps each chunk small enough for focused retrieval while preserving neighboring context across section boundaries.

**Embedding model:** `all-MiniLM-L6-v2` from `sentence-transformers` is used locally. It is fast, inexpensive, and good enough for small policy/report/manual documents.

**Vector database:** Chroma is used through LangChain's vector store wrapper as a local persistent vector store under `data/vector_store`. It is simple to run for a machine task and does not require external infrastructure.

**Retrieval:** The retriever combines Chroma semantic similarity with a lightweight BM25 keyword scorer over stored chunks. Semantic candidates and keyword candidates are merged with a weighted score, which improves exact-term queries while keeping semantic matching for natural-language questions. Responses include document name, page when available, chunk ID, and score.

**Agent approach:** The agent uses LangChain `create_agent` with three structured tools: `search_documents`, `calculator`, and `current_datetime`. Tool artifacts are converted into a user-visible trace. If no LLM key is configured or the agent call fails, the app falls back to a lightweight rule-based planner so local development still works.

**Memory:** Conversation history is stored in SQLite at `data/chat_history.sqlite3` per `session_id`. Recent turns are included in retrieval and response generation so follow-up questions have context, and previous sessions remain available after server restarts.

## Known Limitations

- Chat history is local to the machine because it is stored in a SQLite file under `data/`.
- The current planner is intentionally simple; a production version could use model-native function calling.
- Retrieval uses a lightweight in-process BM25 implementation. For larger datasets, a dedicated search engine or reranker would be more scalable.
- Uploaded files are copied into local storage; object storage would be better for production.
- The fallback answer is extractive and less polished when no LLM key is configured.
