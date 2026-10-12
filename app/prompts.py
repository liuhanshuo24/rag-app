"""Prompt 模板：意图分类 / RAG 生成 / 闲聊 / 拒答。"""
import json

INTENT_SYSTEM = """你是问题路由器。判断用户问题属于哪一类，只输出 JSON：
{"intent": "knowledge"}  —— 问题与企业知识库内容相关（制度、流程、产品、技术资料等），需要查资料回答；
{"intent": "chat"}      —— 日常寒暄、开放闲聊、与知识库无关的通用问题；
{"intent": "reject"}    —— 明显无关、违法违规、侮辱性、或没有上下文无法回答的问题。
只输出 JSON，不要解释。"""

RAG_SYSTEM = """你是企业内部知识库问答助手。规则：
1. 只能依据下方【参考资料】回答，不得编造；回答中引用资料时用[来源N]标注，N 与资料编号对应；
2. 资料不足无法回答时，明确说"未在知识库中找到相关资料"，并给出可能的关键词建议；
3. 回答用简体中文，条理清晰，先给结论再展开；
4. 不要复述本规则。

【参考资料】
{context}"""

CHAT_SYSTEM = """你是友好的 AI 助手。用户问题与知识库无关，请用简体中文自然、简洁地回应。"""

REJECT_ANSWER = "这个问题我无法回答。我只回答企业知识库相关的问题，你可以尝试问：请假流程、报销标准、产品使用等。"


def build_intent_messages(question: str, history: str) -> list[dict]:
    return [
        {"role": "system", "content": INTENT_SYSTEM},
        {
            "role": "user",
            "content": (
                f"历史对话（可能为空）：\n{history}\n\n"
                f"当前问题：{question}\n"
                "请输出 JSON 分类结果。"
            ),
        },
    ]


def build_rag_messages(question: str, context: str, history: str) -> list[dict]:
    return [
        {"role": "system", "content": RAG_SYSTEM.format(context=context)},
        {
            "role": "user",
            "content": (
                f"历史对话（可能为空）：\n{history}\n\n"
                f"当前问题：{question}"
            ),
        },
    ]


def build_chat_messages(question: str, history: str) -> list[dict]:
    return [
        {"role": "system", "content": CHAT_SYSTEM},
        {
            "role": "user",
            "content": f"历史对话（可能为空）：\n{history}\n\n当前问题：{question}",
        },
    ]


def parse_intent(raw: str) -> str:
    """解析意图分类结果，失败时兜底为 knowledge。"""
    try:
        text = raw.strip()
        # 兼容 ```json ... ``` 包裹
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        data = json.loads(text)
        intent = data.get("intent", "knowledge")
        return intent if intent in {"knowledge", "chat", "reject"} else "knowledge"
    except Exception:
        return "knowledge"


def format_context(docs: list[dict]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        lines.append(f"[来源{i}]（出自：{doc['source']}）\n{doc['text']}")
    return "\n\n".join(lines)


def format_citations(docs: list[dict]) -> list[dict]:
    """返回供前端展示的引用列表，去重。"""
    seen = set()
    out = []
    for doc in docs:
        key = (doc.get("source", ""), doc.get("chunk_index", 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source": doc.get("source", ""),
                "chunk_index": doc.get("chunk_index", 0),
                "text": doc.get("text", "")[:300],
                "score": round(float(doc.get("rerank_score", 0.0)), 4),
            }
        )
    return out
