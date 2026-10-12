"""评估脚本：python scripts/evaluate.py [--top-k 5]

对 test_queries.json 中的每个问题执行混合检索，统计：
- Top-K 检索命中率：检索结果中是否包含 expected 中的关键词（任一命中即算该题命中）；
- 平均命中片段数：每题命中 expected 关键词的片段数量。

用法：先用 scripts/ingest.py 建好索引，再运行本脚本。指标可直接用于简历（真实数据）。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retriever import get_retriever  # noqa: E402


def load_queries(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def hit(query: str, expected: list[str], top_k: int) -> tuple[bool, int]:
    docs = get_retriever().retrieve(query, top_k=top_k)
    hit_count = 0
    for d in docs:
        text = d["text"]
        if any(kw in text for kw in expected):
            hit_count += 1
    return hit_count > 0, hit_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    qpath = Path(__file__).resolve().parent / "test_queries.json"
    queries = load_queries(qpath)
    if not queries:
        print("测试集为空，请先编辑 scripts/test_queries.json")
        return

    total = len(queries)
    hit_tasks = 0
    total_hits = 0
    print(f"共 {total} 个测试问题，Top-{args.top_k} 检索评估：\n")

    for q in queries:
        ok, n = hit(q["question"], q["expected"], args.top_k)
        total_hits += n
        if ok:
            hit_tasks += 1
        print(f"[{'✓' if ok else '✗'}] {q['question']}")

    precision = hit_tasks / total
    avg_hits = total_hits / total
    print(f"\n===== 评估结果 =====")
    print(f"Top-{args.top_k} 命中率：{precision * 100:.1f}%  ({hit_tasks}/{total})")
    print(f"平均命中片段数：{avg_hits:.2f}")
    print(f"（可写入简历的指标示例：Top-{args.top_k} 检索准确率 {precision * 100:.1f}%）")


if __name__ == "__main__":
    main()
