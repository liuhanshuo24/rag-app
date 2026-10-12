"""会话记忆：内存版 session → 消息历史，支持多轮上下文拼接与长度截断。"""
from collections import defaultdict

from .config import Config

# session_id -> [(role, content), ...]
_sessions: dict[str, list[tuple[str, str]]] = defaultdict(list)

MAX_TURNS = Config.MEMORY_TURNS
MAX_CHARS = Config.MAX_HISTORY_CHARS


def add_turn(session_id: str, question: str, answer: str) -> None:
    _sessions[session_id].append(("user", question))
    _sessions[session_id].append(("assistant", answer))
    # 只保留最近 N 轮（N 轮 = 2N 条消息）
    keep = MAX_TURNS * 2
    if len(_sessions[session_id]) > keep:
        _sessions[session_id] = _sessions[session_id][-keep:]


def get_history_text(session_id: str) -> str:
    """拼接历史对话文本（带角色标识），超长截断。"""
    lines = []
    total = 0
    for role, content in _sessions.get(session_id, []):
        line = f"{'用户' if role == 'user' else '助手'}: {content}"
        total += len(line)
        if total > MAX_CHARS:
            break
        lines.append(line)
    return "\n".join(lines)


def clear(session_id: str) -> None:
    _sessions.pop(session_id, None)
