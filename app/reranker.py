"""重排模型层：bge-reranker（CrossEncoder），对候选片段按相关性精排。"""
from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import Config


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(
        Config.RERANKER_MODEL,
        device=Config.DEVICE,
        max_length=512,
    )


def rerank_docs(query: str, docs: list[dict], top_k: int | None = None) -> list[dict]:
    """对候选片段重排，返回按相关性降序的 docs（带 rerank_score 字段）。"""
    if not docs:
        return []
    pairs = [[query, d["text"]] for d in docs]
    scores = get_reranker().predict(pairs)
    ranked = []
    for doc, score in zip(docs, scores):
        item = dict(doc)
        item["rerank_score"] = float(score)
        ranked.append(item)
    ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    if top_k:
        ranked = ranked[:top_k]
    return ranked
