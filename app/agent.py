"""Agent 编排层：LangGraph 状态图 —— 意图分类 → 混合检索 → RAG 生成 / 闲聊 / 拒答。

设计亮点（简历可写）：
- 状态图把"意图路由 + 检索 + 生成 + 兜底"显式建模，每个节点输出决策步骤（steps），
  前端可渲染 Agent 的完整决策路径；
- 检索结果为空或置信度低于阈值时自动拒答，避免幻觉；
- 节点间只通过 State 通信，便于替换节点（如换成工具调用、多轮反思）。
"""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .config import Config
from .llm import get_llm
from .memory import get_history_text
from .prompts import (
    REJECT_ANSWER,
    build_chat_messages,
    build_intent_messages,
    build_rag_messages,
    format_citations,
    format_context,
    parse_intent,
)
from .retriever import get_retriever


class AgentState(TypedDict, total=False):
    session_id: str
    question: str
    history: str
    intent: str
    docs: list[dict]
    answer: str
    citations: list[dict]
    confidence: float
    steps: list[dict]


def _step(steps: list[dict], name: str, detail: str) -> None:
    steps.append({"step": name, "detail": detail})


def classify(state: AgentState) -> dict:
    steps = list(state.get("steps", []))
    llm = get_llm()
    history = get_history_text(state["session_id"])
    raw = llm.complete(build_intent_messages(state["question"], history), temperature=0)
    intent = parse_intent(raw)
    _step(steps, "意图分类", f"识别为「{'知识问答' if intent == 'knowledge' else '闲聊' if intent == 'chat' else '拒答'}」")
    return {"intent": intent, "history": history, "steps": steps}


def retrieve(state: AgentState) -> dict:
    steps = list(state.get("steps", []))
    docs = get_retriever().retrieve(state["question"])
    _step(steps, "混合检索", f"BM25+向量融合后重排，召回 {len(docs)} 个片段")
    if docs:
        _step(steps, "重排序", f"最高相关分 {docs[0].get('rerank_score', 0):.3f}")
    return {"docs": docs, "steps": steps}


def generate(state: AgentState) -> dict:
    steps = list(state.get("steps", []))
    docs = state.get("docs", [])
    if not docs or docs[0].get("rerank_score", 0) < Config.REJECT_THRESHOLD:
        _step(steps, "兜底", "检索置信度不足，拒答以防幻觉")
        return {
            "answer": REJECT_ANSWER,
            "citations": [],
            "confidence": 0.0,
            "steps": steps,
        }
    context = format_context(docs)
    messages = build_rag_messages(state["question"], context, state.get("history", ""))
    answer = get_llm().complete(messages)
    _step(steps, "生成回答", f"基于 {len(docs)} 条资料生成，引用已标注")
    return {
        "answer": answer,
        "citations": format_citations(docs),
        "confidence": float(docs[0].get("rerank_score", 0.0)),
        "steps": steps,
    }


def chat_generate(state: AgentState) -> dict:
    steps = list(state.get("steps", []))
    messages = build_chat_messages(state["question"], state.get("history", ""))
    answer = get_llm().complete(messages)
    _step(steps, "闲聊直答", "未检索知识库")
    return {"answer": answer, "citations": [], "confidence": 0.0, "steps": steps}


def reject_node(state: AgentState) -> dict:
    steps = list(state.get("steps", []))
    _step(steps, "拒答", "问题类型不可回答，直接结束")
    return {
        "answer": REJECT_ANSWER,
        "citations": [],
        "confidence": 0.0,
        "steps": steps,
    }


def route(state: AgentState) -> str:
    return state.get("intent", "knowledge")


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("chat_generate", chat_generate)
    g.add_node("reject_node", reject_node)
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        route,
        {
            "knowledge": "retrieve",
            "chat": "chat_generate",
            "reject": "reject_node",
        },
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    g.add_edge("chat_generate", END)
    g.add_edge("reject_node", END)
    return g.compile()


# 复用同一张图（进程内单例）
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_agent(session_id: str, question: str) -> dict:
    """非流式完整推理。"""
    init: AgentState = {"session_id": session_id, "question": question, "steps": []}
    return get_graph().invoke(init)
