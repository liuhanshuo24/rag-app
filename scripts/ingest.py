"""建索引脚本：python scripts/ingest.py
将 data/ 目录下的 .md/.txt/.rst 文档切分并写入 ChromaDB（幂等，可重复执行，新文档自动增量）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexer import build_or_update  # noqa: E402
from app.retriever import get_retriever  # noqa: E402


def main() -> None:
    print("开始构建知识库索引…")
    stats = build_or_update()
    print(f"完成：处理文档 {stats['docs']} 个，写入片段 {stats['added_chunks']} 条")
    get_retriever().refresh_bm25()
    print(f"向量库当前片段总数：{get_retriever()._count()}")


if __name__ == "__main__":
    main()
