"""Non-destructive text-extraction quality signals for ingestion."""

from __future__ import annotations

import re
from typing import Any

from src.core.types import Chunk


# A run this long is unlikely to be a normal English word and commonly comes
# from PDF extraction dropping spaces (for example, ``systemoutputshouldbe``).
_CONCATENATED_RUN = re.compile(r"\b[a-z]{20,}\b", re.IGNORECASE)


def assess_text_quality(text: str, max_samples: int = 5) -> dict[str, Any]:
    """Measure probable missing-space corruption without altering text."""
    runs = _CONCATENATED_RUN.findall(text or "")
    alphabetic_characters = sum(character.isalpha() for character in text or "")
    affected_characters = sum(len(run) for run in runs)
    ratio = affected_characters / alphabetic_characters if alphabetic_characters else 0.0
    return {
        "concatenated_run_count": len(runs),
        "concatenated_character_ratio": round(ratio, 6),
        "samples": runs[:max_samples],
        "needs_review": bool(runs) and ratio >= 0.005,
    }


def assess_chunk_quality(chunks: list[Chunk], target_size: int) -> dict[str, Any]:
    """Audit retrieval-unit length and extraction quality after splitting."""
    lengths = [len(chunk.text or "") for chunk in chunks]
    short_limit = max(80, int(target_size * 0.15))
    long_limit = int(target_size * 1.5)
    flagged = []
    for chunk, length in zip(chunks, lengths):
        text_quality = assess_text_quality(chunk.text)
        quality = {
            "char_length": length,
            "too_short": length < short_limit,
            "too_long": length > long_limit,
            "concatenated_character_ratio": text_quality["concatenated_character_ratio"],
            "needs_review": text_quality["needs_review"],
        }
        chunk.metadata["chunk_quality"] = quality
        if quality["too_short"] or quality["too_long"] or quality["needs_review"]:
            flagged.append(chunk.id)
    return {
        "target_size": target_size,
        "chunk_count": len(chunks),
        "min_char_length": min(lengths, default=0),
        "max_char_length": max(lengths, default=0),
        "avg_char_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "too_short_count": sum(length < short_limit for length in lengths),
        "too_long_count": sum(length > long_limit for length in lengths),
        "needs_review_count": len(flagged),
        "flagged_chunk_ids": flagged[:10],
    }
