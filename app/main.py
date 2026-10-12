"""FastAPI 后端：聊天（非流式/SSE 流式）、知识库上传增量、会话管理、健康检查。"""
import json
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import agent, indexer, memory
from .config import Config
from .llm import get_llm
from .prompts import (
    REJECT_ANSWER,
    build_rag_messages,
    format_citations,
    format_context,
    parse_intent,
)
from .retriever import get_retriever

app = FastAPI(title="RAG 全栈问答助手", version="1.0.0")

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
UPLOAD_DIR = Config.DATA_DIR / "uploads"


class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str


class UploadResult(BaseModel):
    ok: bool
    added_chunks: int = 0
    message: str = ""


# ---------- 静态页面 ----------
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- 健康检查 ----------
@app.get("/api/health")
def health():
    retriever = get_retriever()
    try:
        llm = get_llm()
        llm_provider = llm.provider
        llm_model = llm.model
    except Exception:
        llm_provider = "未配置（请填写 .env 中的 LLM_API_KEY）"
        llm_model = Config.LLM_MODEL
    return {
        "status": "ok",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "kb_chunks": retriever._count(),
        "embedding_model": Config.EMBEDDING_MODEL,
        "reranker_model": Config.RERANKER_MODEL,
    }


@app.get("/api/kb")
def kb_stats():
    retriever = get_retriever()
    return {"kb_chunks": retriever._count(), "data_dir": str(Config.DATA_DIR)}


# ---------- 非流式聊天 ----------
@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    result = agent.run_agent(session_id, question)
    memory.add_turn(session_id, question, result.get("answer", ""))

    return {
        "session_id": session_id,
        "answer": result.get("answer", ""),
        "intent": result.get("intent", "knowledge"),
        "confidence": result.get("confidence", 0.0),
        "citations": result.get("citations", []),
        "steps": result.get("steps", []),
    }


# ---------- SSE 流式聊天 ----------
def _stream_pipeline(session_id: str, question: str):
    """内部流式生成：先跑意图分类+检索，再流式输出答案，最后推送决策信息。"""
    try:
        llm = get_llm()
        history = memory.get_history_text(session_id)
        raw_intent = llm.complete(
            [{"role": "system", "content": "你是问题路由器。判断用户问题属于哪一类，只输出JSON：{\"intent\": \"knowledge\"}知识问答 / {\"intent\": \"chat\"}闲聊 / {\"intent\": \"reject\"}不可回答。只输出JSON。"},
             {"role": "user", "content": f"历史对话：\n{history}\n\n当前问题：{question}"}],
            temperature=0,
        )
        intent = parse_intent(raw_intent)
        steps = [{"step": "意图分类", "detail": f"识别为「{'知识问答' if intent == 'knowledge' else '闲聊' if intent == 'chat' else '拒答'}」"}]

        yield {"event": "intent", "data": json.dumps({"intent": intent})}

        if intent == "reject":
            yield {"event": "delta", "data": json.dumps({"text": REJECT_ANSWER})}
            yield {"event": "steps", "data": json.dumps({"steps": steps})}
            yield {"event": "done", "data": json.dumps({"answer": REJECT_ANSWER})}
            return

        if intent == "chat":
            messages = [
                {"role": "system", "content": "你是友好的 AI 助手。请用简体中文自然、简洁地回应。"},
                {"role": "user", "content": f"历史对话（可能为空）：\n{history}\n\n当前问题：{question}"},
            ]
            answer_parts: list[str] = []
            for token in llm.stream_complete(messages):
                answer_parts.append(token)
                yield {"event": "delta", "data": json.dumps({"text": token})}
            answer = "".join(answer_parts)
            steps.append({"step": "闲聊直答", "detail": "未检索知识库"})
            yield {"event": "steps", "data": json.dumps({"steps": steps})}
            yield {"event": "done", "data": json.dumps({"answer": answer, "citations": []})}
            memory.add_turn(session_id, question, answer)
            return

        # knowledge：混合检索
        docs = get_retriever().retrieve(question)
        steps.append({"step": "混合检索", "detail": f"BM25+向量融合后重排，召回 {len(docs)} 个片段"})
        if docs:
            steps.append({"step": "重排序", "detail": f"最高相关分 {docs[0].get('rerank_score', 0):.3f}"})

        if not docs or docs[0].get("rerank_score", 0) < Config.REJECT_THRESHOLD:
            steps.append({"step": "兜底", "detail": "检索置信度不足，拒答以防幻觉"})
            yield {"event": "delta", "data": json.dumps({"text": REJECT_ANSWER})}
            yield {"event": "steps", "data": json.dumps({"steps": steps})}
            yield {"event": "done", "data": json.dumps({"answer": REJECT_ANSWER, "citations": []})}
            memory.add_turn(session_id, question, REJECT_ANSWER)
            return

        context = format_context(docs)
        messages = build_rag_messages(question, context, history)
        steps.append({"step": "生成回答", "detail": f"基于 {len(docs)} 条资料生成，引用已标注"})
        answer_parts = []
        for token in llm.stream_complete(messages):
            answer_parts.append(token)
            yield {"event": "delta", "data": json.dumps({"text": token})}
        answer = "".join(answer_parts)
        citations = format_citations(docs)
        confidence = float(docs[0].get("rerank_score", 0.0))
        yield {"event": "steps", "data": json.dumps({"steps": steps})}
        yield {"event": "done", "data": json.dumps({"answer": answer, "citations": citations, "confidence": confidence})}
        memory.add_turn(session_id, question, answer)
    except Exception as exc:  # noqa: BLE001
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    session_id = req.session_id or uuid.uuid4().hex
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    return EventSourceResponse(_stream_pipeline(session_id, question))


# ---------- 知识库增量更新 ----------
@app.post("/api/upload", response_model=UploadResult)
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".md", ".txt", ".rst"}:
        raise HTTPException(status_code=400, detail="仅支持 .md / .txt / .rst 文档")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # 防路径穿越：只取文件名
    target = UPLOAD_DIR / Path(file.filename).name
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    stats = indexer.build_or_update()
    # 刷新 BM25 缓存（检索层有 lru_cache，这里主动重置）
    try:
        get_retriever().refresh_bm25()
    except Exception:
        pass

    return UploadResult(ok=True, added_chunks=stats["added_chunks"], message=f"已入库：{file.filename}")


# ---------- 会话 ----------
@app.post("/api/session/clear")
def clear_session(req: ChatRequest):
    if req.session_id:
        memory.clear(req.session_id)
    return {"ok": True}
