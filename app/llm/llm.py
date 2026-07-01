from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

from app.core.config import settings
from app.core.logging import logger

try:
    import openai
except Exception:
    openai = None


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        raise NotImplementedError


def _token_usage(text: str) -> Dict[str, int]:
    tokens = len(text.split())
    return {"prompt_tokens": tokens, "completion_tokens": max(1, tokens // 2), "total_tokens": tokens + max(1, tokens // 2)}


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.openai_api_key
        if openai is not None and self.api_key:
            openai.api_key = self.api_key

    def generate(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        if openai is None or not self.api_key:
            return {"answer": "OpenAI API not configured", "citations": [], "token_usage": _token_usage(prompt)}
        return {"answer": f"OpenAI placeholder response to: {prompt}", "citations": [], "token_usage": _token_usage(prompt)}


class GeminiLLM(BaseLLM):
    def generate(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        return {"answer": f"Gemini placeholder response to: {prompt}", "citations": [], "token_usage": _token_usage(prompt)}


class DummyLLM(BaseLLM):
    def generate(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        if not context:
            return {"answer": "I could not find relevant information in the supplied documents.", "citations": [], "token_usage": _token_usage(prompt)}
        return {"answer": f"Grounded answer using {len(context)} chunks", "citations": [c.get("chunk_id") for c in context if isinstance(c, dict)], "token_usage": _token_usage(prompt)}
