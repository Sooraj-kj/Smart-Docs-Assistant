from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
CHAT_DB_PATH = DATA_DIR / "chat_history.sqlite3"

COLLECTION_NAME = "smart_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_TOP_K = 5

for directory in (DATA_DIR, UPLOAD_DIR, VECTOR_STORE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
