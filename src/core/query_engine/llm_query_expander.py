"""Optional, failure-safe LLM query rewriting for retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.libs.llm.base_llm import BaseLLM, Message


@dataclass(frozen=True)
class LLMQueryExpanderConfig:
    """Limits for a short retrieval rewrite, not an answer generation."""

    max_tokens: int = 96
    timeout_seconds: float = 8.0
    max_rewrite_characters: int = 600


@dataclass(frozen=True)
class QueryExpansion:
    """Outcome of an expansion attempt, including safe fallback metadata."""

    rewrite: str | None
    elapsed_ms: float
    used_fallback: bool
    fallback_reason: str | None = None


class LLMQueryExpander:
    """Generate one compact retrieval rewrite and never block retrieval on failure."""

    _SYSTEM_PROMPT = (
        "Rewrite a search query for document retrieval. Preserve its intent and "
        "add only strongly implied concepts that improve lexical and semantic "
        "matching. Prefer concrete terms from the query domain over vague abstractions. "
        "Output exactly one short search query, with no answer, preamble, bullets, "
        "quotes, or reasoning."
    )

    def __init__(self, llm: BaseLLM, config: LLMQueryExpanderConfig | None = None):
        self.llm = llm
        self.config = config or LLMQueryExpanderConfig()

    def expand(self, query: str) -> QueryExpansion:
        """Return a rewrite, or a structured fallback when the provider fails."""
        started = time.monotonic()
        try:
            response = self.llm.chat(
                [
                    Message(role="system", content=self._SYSTEM_PROMPT),
                    Message(role="user", content=query),
                ],
                temperature=0,
                max_tokens=self.config.max_tokens,
                timeout=self.config.timeout_seconds,
            )
            rewrite = self._clean_rewrite(response.content, query)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            if rewrite is None:
                return QueryExpansion(
                    rewrite=None,
                    elapsed_ms=elapsed_ms,
                    used_fallback=True,
                    fallback_reason="empty_or_invalid_rewrite",
                )
            return QueryExpansion(rewrite=rewrite, elapsed_ms=elapsed_ms, used_fallback=False)
        except Exception as exc:
            return QueryExpansion(
                rewrite=None,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                used_fallback=True,
                fallback_reason=type(exc).__name__,
            )

    def _clean_rewrite(self, content: Any, original_query: str) -> str | None:
        if not isinstance(content, str):
            return None
        rewrite = " ".join(content.strip().split())
        if (
            not rewrite
            or len(rewrite) > self.config.max_rewrite_characters
            or rewrite.casefold() == original_query.strip().casefold()
        ):
            return None
        return rewrite
