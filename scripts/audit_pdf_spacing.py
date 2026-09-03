#!/usr/bin/env python
"""Offline audit and deterministic repair for one PDF's missing-space text."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.settings import load_settings
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.text_quality import assess_chunk_quality, assess_text_quality
from src.ingestion.transform.text_spacing_repair import (
    TextSpacingRepairConfig,
    TextSpacingRepairer,
)
from src.libs.loader.pdf_loader import PdfLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and optionally repair PDF text missing spaces offline."
    )
    parser.add_argument("--pdf", required=True, help="PDF path to audit.")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument(
        "--output-dir",
        default="data/benchmark/results/pdf_spacing_audit",
        help="Directory for raw/repaired text and JSON logs.",
    )
    parser.add_argument(
        "--apply-repair",
        action="store_true",
        help="Write repaired text using the deterministic repairer.",
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _chunk_manifest(chunks: list) -> list[dict]:
    rows = []
    for chunk in chunks:
        source_path = str(chunk.metadata.get("source_path", ""))
        chunk_index = chunk.metadata.get("chunk_index")
        source_hash = (
            hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
            if source_path and isinstance(chunk_index, int)
            else None
        )
        rows.append(
            {
                "chunk_id": chunk.id,
                "stable_chunk_key": (
                    f"{source_hash}_{chunk_index:04d}"
                    if source_hash is not None
                    else None
                ),
                "chunk_index": chunk_index,
                "char_length": len(chunk.text),
                "chunk_quality": chunk.metadata.get("chunk_quality"),
                "text": chunk.text,
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    settings = load_settings(args.config)
    pdf_path = Path(args.pdf)
    output_dir = Path(args.output_dir) / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = PdfLoader(extract_images=False)
    document = loader.load(pdf_path)
    raw_text = document.text
    raw_quality = assess_text_quality(raw_text, max_samples=20)

    repair_settings = (
        getattr(settings.ingestion, "text_spacing_repair", None)
        if settings.ingestion is not None
        else None
    ) or {}
    repairer = TextSpacingRepairer(
        TextSpacingRepairConfig(
            enabled=args.apply_repair,
            min_run_length=int(repair_settings.get("min_run_length", 20)),
            max_run_length=int(repair_settings.get("max_run_length", 120)),
            max_repairs=int(repair_settings.get("max_repairs", 1000)),
        )
    )
    repair_result = repairer.repair(raw_text)
    repaired_quality = assess_text_quality(repair_result.text, max_samples=20)

    chunker = DocumentChunker(settings)
    raw_chunks = chunker.split_document(document)
    raw_chunk_quality = assess_chunk_quality(
        raw_chunks,
        settings.ingestion.chunk_size if settings.ingestion else 1000,
    )
    repaired_document = document.__class__(
        id=document.id,
        text=repair_result.text,
        metadata=document.metadata.copy(),
    )
    repaired_chunks = chunker.split_document(repaired_document)
    repaired_chunk_quality = assess_chunk_quality(
        repaired_chunks,
        settings.ingestion.chunk_size if settings.ingestion else 1000,
    )

    (output_dir / "raw.txt").write_text(raw_text, encoding="utf-8")
    (output_dir / "repaired.txt").write_text(repair_result.text, encoding="utf-8")
    _write_json(output_dir / "raw_chunks.json", _chunk_manifest(raw_chunks))
    _write_json(output_dir / "repaired_chunks.json", _chunk_manifest(repaired_chunks))
    _write_jsonl(output_dir / "spacing_repair_log.jsonl", repair_result.repairs)
    _write_jsonl(output_dir / "spacing_repair_skipped.jsonl", repair_result.skipped)
    _write_json(
        output_dir / "quality_before_after.json",
        {
            "pdf": str(pdf_path),
            "doc_id": document.id,
            "repair_enabled": args.apply_repair,
            "changed": repair_result.changed,
            "repair_count": len(repair_result.repairs),
            "skipped_count": len(repair_result.skipped),
            "text_length_before": len(raw_text),
            "text_length_after": len(repair_result.text),
            "text_quality_before": raw_quality,
            "text_quality_after": repaired_quality,
            "chunk_quality_before": raw_chunk_quality,
            "chunk_quality_after": repaired_chunk_quality,
        },
    )

    print(f"PDF: {pdf_path}")
    print(f"Output dir: {output_dir}")
    print(
        "Text quality ratio: "
        f"{raw_quality['concatenated_character_ratio']:.4f} -> "
        f"{repaired_quality['concatenated_character_ratio']:.4f}"
    )
    print(f"Repairs accepted: {len(repair_result.repairs)}")
    print(f"Runs skipped: {len(repair_result.skipped)}")
    print(
        "Chunks needing review: "
        f"{raw_chunk_quality['needs_review_count']} -> "
        f"{repaired_chunk_quality['needs_review_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
