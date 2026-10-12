"""全局配置：从项目根目录 .env 读取，未配置时使用内置默认值。"""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Config:
    # ---------- LLM（可插拔） ----------
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # ---------- 嵌入 / 重排 ----------
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    DEVICE = os.getenv("DEVICE", "cpu")

    # ---------- 检索 ----------
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
    TOP_K = int(os.getenv("TOP_K", "10"))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))

    # 拒答置信度阈值：检索重排后最高分低于该值则拒答
    REJECT_THRESHOLD = float(os.getenv("REJECT_THRESHOLD", "0.35"))

    # ---------- 路径 ----------
    DATA_DIR = BASE_DIR / os.getenv("DATA_DIR", "data")
    SAMPLE_DIR = BASE_DIR / "data" / "sample"
    CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kb_chunks")

    # ---------- 会话记忆 ----------
    MEMORY_TURNS = int(os.getenv("MEMORY_TURNS", "4"))
    MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "2000"))
