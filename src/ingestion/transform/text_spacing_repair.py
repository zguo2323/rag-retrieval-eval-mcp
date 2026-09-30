"""Conservative repair for PDF text extraction that drops English spaces."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


_CONCATENATED_RUN = re.compile(r"\b[a-z]{20,}\b", re.IGNORECASE)
_SKIP_PATTERN = re.compile(r"(https?://|www\.|[_/@\\]|\d)")


_COMMON_WORDS = {
    "a",
    "ability",
    "able",
    "abstraction",
    "abstractions",
    "accurately",
    "about",
    "across",
    "adapted",
    "add",
    "advanced",
    "ai",
    "algorithm",
    "algorithms",
    "all",
    "also",
    "an",
    "and",
    "another",
    "are",
    "as",
    "audience",
    "audiences",
    "available",
    "be",
    "because",
    "behaviors",
    "best",
    "between",
    "by",
    "can",
    "case",
    "certain",
    "clear",
    "communication",
    "comparison",
    "concept",
    "concepts",
    "commercial",
    "confidence",
    "context",
    "contextualize",
    "could",
    "data",
    "decision",
    "decisions",
    "detailed",
    "detect",
    "diagnostic",
    "different",
    "differ",
    "differences",
    "domain",
    "each",
    "effect",
    "explain",
    "explainability",
    "explanation",
    "explanations",
    "explainable",
    "experimental",
    "from",
    "for",
    "given",
    "groups",
    "human",
    "humans",
    "employ",
    "if",
    "implies",
    "in",
    "individual",
    "information",
    "input",
    "interpretation",
    "interpretations",
    "interpretable",
    "is",
    "it",
    "learning",
    "led",
    "less",
    "machine",
    "manner",
    "materials",
    "maybe",
    "meaningful",
    "mechanism",
    "mental",
    "model",
    "models",
    "more",
    "must",
    "national",
    "necessarily",
    "need",
    "needs",
    "of",
    "one",
    "or",
    "other",
    "output",
    "outputs",
    "paradigms",
    "people",
    "personality",
    "precise",
    "preference",
    "preferences",
    "procedure",
    "provider",
    "purpose",
    "reason",
    "reasoning",
    "related",
    "risk",
    "risks",
    "representation",
    "representations",
    "review",
    "rely",
    "science",
    "should",
    "shows",
    "skills",
    "so",
    "some",
    "such",
    "system",
    "systematic",
    "systematically",
    "systems",
    "tailor",
    "tailored",
    "technology",
    "terms",
    "theory",
    "that",
    "the",
    "their",
    "therefore",
    "this",
    "to",
    "trace",
    "trust",
    "types",
    "use",
    "user",
    "users",
    "versus",
    "ways",
    "when",
    "why",
    "with",
}


@dataclass(frozen=True)
class TextSpacingRepairConfig:
    """Runtime limits for deterministic text spacing repair."""

    enabled: bool = False
    min_run_length: int = 20
    max_run_length: int = 120
    max_repairs: int = 1000


@dataclass(frozen=True)
class TextSpacingRepairResult:
    """Output from one repair pass."""

    text: str
    repairs: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.repairs)


class TextSpacingRepairer:
    """Split obvious missing-space English runs without using LLM calls.

    The repairer is intentionally conservative: it only accepts segmentations
    fully covered by a small local lexicon and skips IDs, URLs, numeric runs,
    and ambiguous leftovers. Unknown text is preserved exactly.
    """

    def __init__(self, config: TextSpacingRepairConfig | None = None) -> None:
        self.config = config or TextSpacingRepairConfig()
        self._words = _COMMON_WORDS

    def repair(self, text: str) -> TextSpacingRepairResult:
        """Repair likely missing spaces and return an auditable log."""
        if not self.config.enabled or not text:
            return TextSpacingRepairResult(text=text or "")

        repairs: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        words = self._words | self._context_words(text)
        max_word_length = max(map(len, words))

        def replace(match: re.Match[str]) -> str:
            original = match.group(0)
            reason = self._skip_reason(original)
            if reason is not None:
                skipped.append(self._log_item(match, original, None, reason))
                return original
            if len(repairs) >= self.config.max_repairs:
                skipped.append(self._log_item(match, original, None, "max_repairs_reached"))
                return original
            split_words = self._camel_case_words(original, words)
            if split_words is None:
                split_words = self._segment(original.lower(), words, max_word_length)
            if split_words is None:
                skipped.append(self._log_item(match, original, None, "no_confident_segmentation"))
                return original
            repaired = self._match_case(original, " ".join(split_words))
            repairs.append(self._log_item(match, original, repaired, "accepted"))
            return repaired

        repaired_text = _CONCATENATED_RUN.sub(replace, text)
        return TextSpacingRepairResult(text=repaired_text, repairs=repairs, skipped=skipped)

    def _skip_reason(self, token: str) -> str | None:
        if len(token) < self.config.min_run_length:
            return "below_min_run_length"
        if len(token) > self.config.max_run_length:
            return "above_max_run_length"
        if _SKIP_PATTERN.search(token):
            return "contains_identifier_or_url_character"
        return None

    def _segment(
        self, token: str, words: set[str], max_word_length: int
    ) -> list[str] | None:
        best: list[tuple[float, list[str]]] = [(math.inf, [])] + [
            (math.inf, []) for _ in range(len(token))
        ]
        best[0] = (0.0, [])

        for end in range(1, len(token) + 1):
            start_min = max(0, end - max_word_length)
            for start in range(start_min, end):
                word = token[start:end]
                if word not in words:
                    continue
                previous_score, previous_words = best[start]
                if math.isinf(previous_score):
                    continue
                score = previous_score + self._word_cost(word)
                if score < best[end][0]:
                    best[end] = (score, [*previous_words, word])

        words = best[len(token)][1]
        if not self._is_confident(words, token):
            return None
        return words

    def _camel_case_words(self, token: str, words: set[str]) -> list[str] | None:
        if not re.search(r"[a-z][A-Z]", token):
            return None
        parts = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", token).split()
        if len(parts) < 2:
            return None
        lowered = [part.lower() for part in parts]
        if all(part in words or len(part) <= 3 for part in lowered):
            return lowered
        return None

    def _context_words(self, text: str) -> set[str]:
        words = set()
        max_common_word_length = max(map(len, self._words))
        for match in re.finditer(r"\b[A-Za-z]{2,24}\b", text):
            word = match.group(0).lower()
            if len(word) >= self.config.min_run_length:
                continue
            if self._segment(word, self._words, max_common_word_length) is not None:
                continue
            words.add(word)
        return words

    @staticmethod
    def _word_cost(word: str) -> float:
        # Prefer fewer, longer words while avoiding one-letter fragments.
        return 1.0 / max(len(word), 1)

    @staticmethod
    def _is_confident(words: list[str], token: str) -> bool:
        if len(words) < 2:
            return False
        if "".join(words) != token:
            return False
        if any(len(word) == 1 and word not in {"a"} for word in words):
            return False
        return sum(len(word) for word in words) / len(words) >= 3.0

    @staticmethod
    def _match_case(original: str, repaired_lower: str) -> str:
        if original.isupper():
            return repaired_lower.upper()
        if original[:1].isupper():
            return repaired_lower[:1].upper() + repaired_lower[1:]
        return repaired_lower

    @staticmethod
    def _log_item(
        match: re.Match[str],
        original: str,
        repaired: str | None,
        reason: str,
    ) -> dict[str, Any]:
        return {
            "start": match.start(),
            "end": match.end(),
            "original": original,
            "repaired": repaired,
            "reason": reason,
        }
