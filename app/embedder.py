"""嵌入模型层：基于 sentence-transformers 加载 bge 中文模型（首次运行自动下载）。"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import Config


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(Config.EMBEDDING_MODEL, device=Config.DEVICE)


def embed_texts(texts: list[str]) -> np.ndarray:
    """批量向量化，输出已归一化的 float32 向量。"""
    emb = get_embedder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(emb, dtype="float32")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
