"""索引层：文档加载 → 文本切分 → 向量化 → 写入 ChromaDB。

设计要点：
- 递归切分：优先按段落(\\n\\n)切，过长再按句子/字符切，保证语义完整；
- 幂等 upsert：chunk 的 id = "{source}::{chunk_index}"，重复运行不产生重复数据，
  新文档可直接追加（增量更新）；
- BM25 索引在检索层按需从向量库重建，保证与向量数据一致。
"""
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection

from .config import Config
from .embedder import embed_texts

# 支持的文档后缀（可自行扩展，如 .docx 需额外依赖）
SUPPORTED_SUFFIXES = {".md", ".txt", ".rst"}


def _load_text(path: Path) -> str:
    """按 UTF-8 读取文本文件，失败时尝试常见编码。"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return Path(path).read_text(encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def iter_docs(data_dir: Path):
    """递归遍历 data 目录下的所有支持文档。"""
    if not data_dir.exists():
        return
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def split_text(text: str, chunk_size: int = Config.CHUNK_SIZE, chunk_overlap: int = Config.CHUNK_OVERLAP) -> list[str]:
    """递归字符切分：段落 → 行 → 字符窗，带重叠。"""
    text = text.strip()
    if not text:
        return []

    def _split_long(segment: str) -> list[str]:
        if len(segment) <= chunk_size:
            return [segment]
        # 超过 chunk_size 的长段：按窗口切，重叠 chunk_overlap 个字符
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size
        return [segment[i : i + chunk_size] for i in range(0, len(segment), step)]

    chunks: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for para in paragraphs:
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            # 长段落先按行聚合到接近 chunk_size，再切分
            lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
            buf = ""
            for line in lines:
                if buf and len(buf) + len(line) + 1 > chunk_size:
                    chunks.extend(_split_long(buf))
                    buf = line
                else:
                    buf = f"{buf}\n{line}" if buf else line
            if buf:
                chunks.extend(_split_long(buf))
    return chunks


def get_or_create_collection() -> Collection:
    client = chromadb.PersistentClient(path=str(Config.CHROMA_DIR))
    return client.get_or_create_collection(Config.COLLECTION_NAME)


def build_or_update(data_dir: Path | None = None, collection: Collection | None = None) -> dict:
    """全量/增量写入向量库。重复调用安全：同 id 会被覆盖更新。"""
    collection = collection or get_or_create_collection()
    data_dir = data_dir or Config.DATA_DIR

    added = 0
    for path in iter_docs(data_dir):
        text = _load_text(path)
        chunks = split_text(text)
        rel = path.relative_to(data_dir).as_posix()
        ids = [f"{rel}::{i}" for i in range(len(chunks))]
        metas = [
            {"source": rel, "chunk_index": i, "length": len(c)} for i, c in enumerate(chunks)
        ]
        if not chunks:
            continue
        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=metas,
            embeddings=embed_texts(chunks).tolist(),
        )
        added += len(chunks)
    return {"added_chunks": added, "docs": sum(1 for _ in iter_docs(data_dir))}
