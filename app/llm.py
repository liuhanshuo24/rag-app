"""可插拔大模型层：DeepSeek / Qwen / 豆包 / OpenAI 全部走 OpenAI 兼容协议。

切换模型只需改 .env 里的 LLM_PROVIDER / LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，
无需改任何业务代码 —— 这是本项目"灵活更换大模型"的设计核心。
"""
from typing import Iterator

from openai import OpenAI

from .config import Config

# 各 provider 的默认接入信息（.env 里显式配置时优先）
PROVIDER_PRESETS = {
    "deepseek": {"base_url": "https://api.deepseek.com", "model": "deepseek-flash"},
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-plus",
    },
    "doubao": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-pro-32k",
    },
    "openai": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
}


class LLMProvider:
    """封装一个 OpenAI 兼容的 chat completions 客户端。"""

    def __init__(self) -> None:
        preset = PROVIDER_PRESETS.get(Config.LLM_PROVIDER, {})
        self.provider = Config.LLM_PROVIDER
        self.base_url = Config.LLM_BASE_URL or preset.get("base_url")
        self.model = Config.LLM_MODEL or preset.get("model")
        self.api_key = Config.LLM_API_KEY
        if not self.api_key:
            raise RuntimeError(
                "未配置 LLM_API_KEY，请复制 .env.example 为 .env 并填入你的模型密钥"
            )
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat(
        self,
        messages: list[dict],
        temperature: float | None = None,
        stream: bool = False,
    ):
        """非流式返回完整响应对象；流式返回 token 迭代器。"""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=Config.LLM_TEMPERATURE if temperature is None else temperature,
            stream=stream,
        )

    def complete(self, messages: list[dict], temperature: float | None = None) -> str:
        """一次性返回文本。"""
        resp = self.chat(messages, temperature=temperature)
        return (resp.choices[0].message.content or "").strip()

    def stream_complete(self, messages: list[dict], temperature: float | None = None) -> Iterator[str]:
        """逐 token 产出文本（流式）。"""
        resp = self.chat(messages, temperature=temperature, stream=True)
        for chunk in resp:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# 全局单例（首次使用才真正建立客户端）
_llm: LLMProvider | None = None


def get_llm() -> LLMProvider:
    global _llm
    if _llm is None:
        _llm = LLMProvider()
    return _llm
