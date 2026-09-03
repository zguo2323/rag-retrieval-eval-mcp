#!/usr/bin/env python
"""Validate the manually annotated NIST AI retrieval golden set."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN = REPO_ROOT / "data/benchmark/golden_nist_ai_40.json"
DEFAULT_CHROMA = REPO_ROOT / "data/db/chroma"
DEFAULT_BM25 = REPO_ROOT / "data/db/bm25/nist_ai_governance/nist_ai_governance_bm25.json"
REQUIRED_FIELDS = {
    "id",
    "category",
    "query",
    "answerable",
    "reference_answer",
    "expected_chunk_ids",
    "expected_sources",
    "evidence",
    "difficulty",
    "split",
    "notes",
}
CATEGORY_COUNTS = {
    "exact_term": 10,
    "semantic_paraphrase": 10,
    "cross_document": 10,
    "structured_fact": 5,
    "unanswerable": 5,
}
DIFFICULTIES = {"easy", "medium", "hard"}
SPLITS = {"dev", "test"}


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_quote_text(value: str) -> str:
    """Normalize PDF line wrapping and whitespace without fuzzy word matching."""
    value = unicodedata.normalize("NFKC", value)
    value = value.translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"}))
    value = re.sub(r"(?<=\w)-\s+(?=\w)", "", value)
    return re.sub(r"\s+", "", value).casefold()


def _supporting_text_matches(supporting_text: str, document: str) -> bool:
    """Check ellipsis-separated quote fragments occur in document order."""
    normalized_document = _normalize_quote_text(document)
    fragments = [
        _normalize_quote_text(fragment)
        for fragment in re.split(r"(?:\.{3,}|…)", supporting_text)
        if fragment.strip()
    ]
    if not fragments:
        return False
    offset = 0
    for fragment in fragments:
        position = normalized_document.find(fragment, offset)
        if position < 0:
            return False
        offset = position + len(fragment)
    return True


def _stable_chunk_position_id(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return chunk_id


def _stable_key_for_stored_chunk(chunk_id: str, metadata: dict[str, Any]) -> str:
    stable_key = metadata.get("stable_chunk_key")
    if stable_key:
        return str(stable_key)
    return _stable_chunk_position_id(chunk_id)


def validate(
    golden_path: Path,
    chroma_path: Path,
    bm25_path: Path,
    match_stable_chunk_index: bool = False,
) -> list[str]:
    errors: list[str] = []
    payload: dict[str, Any] = json.loads(golden_path.read_text(encoding="utf-8"))
    cases = payload.get("test_cases")
    _require(isinstance(cases, list), "test_cases must be a list", errors)
    if not isinstance(cases, list):
        return errors

    _require(payload.get("name") == "nist_ai_governance_golden_40", "unexpected top-level name", errors)
    _require(payload.get("version") == "1.0", "unexpected top-level version", errors)
    _require(payload.get("collection") == "nist_ai_governance", "unexpected collection", errors)
    _require(_nonempty_string(payload.get("description")), "description must be a non-empty string", errors)
    annotation_policy = payload.get("annotation_policy")
    _require(isinstance(annotation_policy, dict), "annotation_policy must be an object", errors)
    if isinstance(annotation_policy, dict):
        _require(annotation_policy.get("relevance") == "strict",
                 "annotation_policy.relevance must be strict", errors)
        _require(annotation_policy.get("created_by") == "human-style manual annotation",
                 "unexpected annotation_policy.created_by", errors)
        _require(_nonempty_string(annotation_policy.get("notes")),
                 "annotation_policy.notes must be a non-empty string", errors)
    _require(len(cases) == 40, f"expected 40 cases, found {len(cases)}", errors)

    ids = [case.get("id") for case in cases]
    queries = [case.get("query") for case in cases]
    _require(len(ids) == len(set(ids)), "duplicate case id", errors)
    _require(len(queries) == len(set(queries)), "duplicate query", errors)
    _require(Counter(case.get("category") for case in cases) == CATEGORY_COUNTS,
             f"category counts differ: {Counter(case.get('category') for case in cases)}", errors)
    _require(Counter(case.get("split") for case in cases) == {"dev": 24, "test": 16},
             f"split counts differ: {Counter(case.get('split') for case in cases)}", errors)

    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_collection("nist_ai_governance")
        stored = collection.get(include=["documents", "metadatas"])
        metadata_by_id = dict(zip(stored["ids"], stored["metadatas"]))
        document_by_id = dict(zip(stored["ids"], stored["documents"]))
        metadata_by_stable_key = {
            _stable_key_for_stored_chunk(chunk_id, metadata): metadata
            for chunk_id, metadata in metadata_by_id.items()
        }
        document_by_stable_key = {
            _stable_key_for_stored_chunk(chunk_id, metadata_by_id[chunk_id]): document
            for chunk_id, document in document_by_id.items()
        }
    except Exception as exc:  # pragma: no cover - environment-specific diagnostic
        errors.append(f"could not read Chroma collection: {exc}")
        return errors

    try:
        bm25_payload = json.loads(bm25_path.read_text(encoding="utf-8"))
        _require(bm25_payload.get("metadata", {}).get("collection") == "nist_ai_governance",
                 "unexpected BM25 collection metadata", errors)
        bm25_ids = {
            posting["chunk_id"]
            for term_data in bm25_payload.get("index", {}).values()
            for posting in term_data.get("postings", [])
        }
        bm25_stable_keys = {
            _stable_chunk_position_id(chunk_id) for chunk_id in bm25_ids
        }
    except Exception as exc:
        errors.append(f"could not read BM25 index: {exc}")
        return errors

    split_by_chunk: dict[str, set[str]] = {}

    for position, case in enumerate(cases, start=1):
        label = case.get("id", f"case #{position}")
        missing = REQUIRED_FIELDS - set(case)
        _require(not missing, f"{label}: missing fields {sorted(missing)}", errors)
        _require(isinstance(case.get("id"), str) and bool(re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", case["id"])),
                 f"{label}: id must be non-empty English snake_case", errors)
        _require(isinstance(case.get("query"), str) and bool(case["query"].strip()),
                 f"{label}: query must be a non-empty string", errors)
        _require(case.get("category") in CATEGORY_COUNTS, f"{label}: invalid category", errors)
        _require(case.get("difficulty") in DIFFICULTIES, f"{label}: invalid difficulty", errors)
        _require(case.get("split") in SPLITS, f"{label}: invalid split", errors)
        _require(isinstance(case.get("answerable"), bool), f"{label}: answerable must be bool", errors)

        chunk_ids = case.get("expected_chunk_ids", [])
        sources = case.get("expected_sources", [])
        evidence = case.get("evidence", [])
        _require(isinstance(chunk_ids, list), f"{label}: expected_chunk_ids must be a list", errors)
        _require(isinstance(sources, list), f"{label}: expected_sources must be a list", errors)
        _require(isinstance(evidence, list), f"{label}: evidence must be a list", errors)

        if case.get("answerable") is True:
            _require(bool(chunk_ids), f"{label}: answerable case has no expected chunks", errors)
            _require(_nonempty_string(case.get("reference_answer")),
                     f"{label}: answerable case has empty reference_answer", errors)
            _require({item.get("chunk_id") for item in evidence if isinstance(item, dict)} == set(chunk_ids),
                     f"{label}: evidence IDs do not exactly match expected_chunk_ids", errors)
        elif case.get("answerable") is False:
            _require(chunk_ids == [], f"{label}: unanswerable case has expected chunks", errors)
            _require(sources == [], f"{label}: unanswerable case has expected sources", errors)
            _require(evidence == [], f"{label}: unanswerable case has evidence", errors)
            _require(case.get("reference_answer") is None,
                     f"{label}: unanswerable case has non-null reference_answer", errors)

        _require(len(chunk_ids) == len(set(chunk_ids)),
                 f"{label}: duplicate expected chunk ID within case", errors)
        evidence_ids = [item.get("chunk_id") for item in evidence if isinstance(item, dict)]
        _require(len(evidence_ids) == len(set(evidence_ids)),
                 f"{label}: duplicate evidence chunk ID within case", errors)

        actual_sources: set[str] = set()
        for chunk_id in chunk_ids:
            lookup_id = (
                _stable_chunk_position_id(chunk_id)
                if match_stable_chunk_index
                else chunk_id
            )
            metadata_lookup = (
                metadata_by_stable_key if match_stable_chunk_index else metadata_by_id
            )
            _require(
                lookup_id in metadata_lookup,
                f"{label}: missing Chroma chunk {chunk_id}",
                errors,
            )
            if lookup_id not in metadata_lookup:
                continue
            source_path = metadata_lookup[lookup_id].get("source_path", "")
            actual_sources.add(Path(source_path).name)
            _require(
                lookup_id in (bm25_stable_keys if match_stable_chunk_index else bm25_ids),
                f"{label}: expected chunk missing from BM25: {chunk_id}",
                errors,
            )
            split_by_chunk.setdefault(lookup_id, set()).add(case.get("split"))
        _require(set(sources) == actual_sources,
                 f"{label}: expected_sources {sources} != metadata sources {sorted(actual_sources)}", errors)

        for item in evidence:
            if not isinstance(item, dict):
                errors.append(f"{label}: evidence item must be an object")
                continue
            _require({"chunk_id", "source", "reason", "supporting_text"} <= set(item),
                     f"{label}: malformed evidence item", errors)
            _require(_nonempty_string(item.get("reason")),
                     f"{label}: evidence reason must be a non-empty string", errors)
            _require(_nonempty_string(item.get("supporting_text")),
                     f"{label}: supporting_text must be a non-empty string", errors)
            chunk_id = item.get("chunk_id")
            lookup_id = (
                _stable_chunk_position_id(chunk_id)
                if match_stable_chunk_index
                else chunk_id
            )
            metadata_lookup = (
                metadata_by_stable_key if match_stable_chunk_index else metadata_by_id
            )
            document_lookup = (
                document_by_stable_key if match_stable_chunk_index else document_by_id
            )
            if lookup_id in metadata_lookup:
                actual = Path(metadata_lookup[lookup_id].get("source_path", "")).name
                _require(item.get("source") == actual,
                         f"{label}: evidence source for {chunk_id} is {item.get('source')}, expected {actual}",
                         errors)
                if _nonempty_string(item.get("supporting_text")):
                    _require(
                        _supporting_text_matches(
                            item["supporting_text"], document_lookup[lookup_id]
                        ),
                        f"{label}: supporting_text does not occur in order in chunk {chunk_id}",
                        errors,
                    )

    for split in SPLITS:
        observed = {case["category"] for case in cases if case.get("split") == split}
        _require(observed == set(CATEGORY_COUNTS), f"{split}: does not cover all categories", errors)

    leaked = sorted(chunk_id for chunk_id, splits in split_by_chunk.items() if len(splits) > 1)
    _require(not leaked, f"expected chunks shared by dev and test: {leaked}", errors)

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--chroma", type=Path, default=DEFAULT_CHROMA)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument(
        "--match-stable-chunk-index",
        action="store_true",
        help=(
            "Validate expected chunks by source-path hash and chunk_index, "
            "ignoring content-hash suffixes."
        ),
    )
    args = parser.parse_args()
    errors = validate(
        args.golden.resolve(),
        args.chroma.resolve(),
        args.bm25.resolve(),
        match_stable_chunk_index=args.match_stable_chunk_index,
    )
    if errors:
        print(f"FAIL: {len(errors)} validation error(s)")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "PASS: 40 cases; schema/quotas/Chroma/BM25/source/evidence/cross-split checks succeeded"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
