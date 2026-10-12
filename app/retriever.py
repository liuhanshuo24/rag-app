"""检索层：混合检索（BM25 关键词 + 向量语义）→ RRF 融合 → bge-reranker 精排。

这是项目检索质量的核心：
- BM25 负责"关键词精确命中"（jieba 分词，可加自定义词典）；
- 向量检索负责"语义相似"（bge-large-zh 中文向量）；
- RRF 将两路排名融合，避免单一检索的盲区；
- 重排器对融合后的候选做最终相关性精排，兼顾精度与性能。
"""
from functools import lru_cache

import jieba
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from .config import Config
from .embedder import embed_query
from .indexer import get_or_create_collection
from .reranker import rerank_docs


@lru_cache(maxsize=1)
def _jieba_ready() -> bool:
    # 可在此处加载自定义词典：jieba.load_userdict(str(Config.DATA_DIR / "dict.txt"))
    return True


def _tokenize(text: str) -> list[str]:
    _jieba_ready()
    return [w for w in jieba.cut_for_search(text) if w.strip()]


class HybridRetriever:
    def __init__(self, collection: Collection | None = None) -> None:
        self.collection = collection or get_or_create_collection()
        self._bms_cache: tuple[BM25Okapi, list[dict]] | None = None

    # ---------- BM25 索引（懒构建，数据更新后可调用 refresh_bm25） ----------
    def _bm25(self):
        if self._bms_cache is None:
            self.refresh_bm25()
        return self._bms_cache  # type: ignore

    def refresh_bm25(self) -> None:
        data = self.collection.get(include=["documents", "metadatas"])
        ids: list[str] = data["ids"]
        docs: list[str] = data["documents"] or []
        metas: list[dict] = data["metadatas"] or []
        corpus = [_tokenize(d) for d in docs]
        meta_list = [
            {
                "id": cid,
                "text": doc,
                "source": (m or {}).get("source", ""),
                "chunk_index": (m or {}).get("chunk_index", 0),
            }
            for cid, doc, m in zip(ids, docs, metas)
        ]
        self._bms_cache = (BM25Okapi(corpus), meta_list)

    def _count(self) -> int:
        return self.collection.count()

    # ---------- 两条检索路 ----------
    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        if self._count() == 0:
            return []
        emb = embed_query(query).tolist()
        res = self.collection.query(query_embeddings=[emb], n_results=min(top_k, self._count()))
        out = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0]
        for cid, doc, dist, m in zip(ids, docs, dists, metas):
            out.append(
                {
                    "id": cid,
                    "text": doc,
                    "source": (m or {}).get("source", ""),
                    "chunk_index": (m or {}).get("chunk_index", 0),
                    "vec_dist": float(dist),
                }
            )
        return out

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        bm25, meta_list = self._bm25()
        if not meta_list:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            item = dict(meta_list[i])
            item["bm25_score"] = float(scores[i])
            out.append(item)
        return out

    # ---------- RRF 融合 ----------
    @staticmethod
    def _rrf_fuse(vec: list[dict], bm: list[dict], k: int = Config.RRF_K) -> list[dict]:
        rank_map: dict[str, dict] = {}
        for rank, item in enumerate(vec + bm):
            cid = item["id"]
            score = 1.0 / (k + rank + 1)  # rank 从 0 开始
            if cid in rank_map:
                rank_map[cid]["rrf_score"] += score
            else:
                rank_map[cid] = dict(item)
                rank_map[cid]["rrf_score"] = score
        merged = sorted(rank_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        return merged

    # ---------- 对外主入口 ----------
    def retrieve(
        self,
        query: str,
        top_k: int = Config.TOP_K,
        rerank_top_k: int = Config.RERANK_TOP_K,
    ) -> list[dict]:
        if self._count() == 0:
            return []
        vec = self._vector_search(query, top_k)
        bm = self._bm25_search(query, top_k)
        fused = self._rrf_fuse(vec, bm)
        ranked = rerank_docs(query, fused, top_k=rerank_top_k)
        return ranked


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
