from __future__ import annotations

from src.core.query_engine.llm_query_expander import LLMQueryExpander, LLMQueryExpanderConfig
from src.libs.llm.base_llm import BaseLLM, ChatResponse, Message


class FakeLLM(BaseLLM):
    def __init__(self, content: str = "tailored explanations for users with different skills"):
        self.content = content
        self.kwargs = None

    def chat(self, messages: list[Message], trace=None, **kwargs):
        self.kwargs = kwargs
        return ChatResponse(content=self.content, model="fake")


class FailingLLM(BaseLLM):
    def chat(self, messages: list[Message], trace=None, **kwargs):
        raise TimeoutError("provider timed out")


def test_expander_returns_compact_rewrite_with_bounded_generation() -> None:
    llm = FakeLLM()
    result = LLMQueryExpander(llm, LLMQueryExpanderConfig(max_tokens=42, timeout_seconds=3)).expand(
        "Why adapt explanations for audiences?"
    )

    assert result.rewrite == "tailored explanations for users with different skills"
    assert result.used_fallback is False
    assert llm.kwargs == {"temperature": 0, "max_tokens": 42, "timeout": 3}


def test_expander_falls_back_on_provider_error() -> None:
    result = LLMQueryExpander(FailingLLM()).expand("original query")

    assert result.rewrite is None
    assert result.used_fallback is True
    assert result.fallback_reason == "TimeoutError"


def test_expander_rejects_empty_or_unchanged_rewrites() -> None:
    result = LLMQueryExpander(FakeLLM(" original query ")).expand("original query")

    assert result.rewrite is None
    assert result.used_fallback is True
    assert result.fallback_reason == "empty_or_invalid_rewrite"
