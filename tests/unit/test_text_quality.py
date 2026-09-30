from src.core.types import Chunk
from src.ingestion.transform.text_quality import assess_chunk_quality, assess_text_quality


def test_detects_pdf_style_concatenated_runs_without_changing_text():
    text = "Thisimpliesthatsystemoutputshouldbetailoredtodifferenttypesofusers."

    quality = assess_text_quality(text)

    assert quality["concatenated_run_count"] == 1
    assert quality["needs_review"] is True
    assert "systemoutput" in quality["samples"][0]


def test_normal_text_is_not_flagged():
    quality = assess_text_quality("System output should be tailored to different users.")

    assert quality["concatenated_run_count"] == 0
    assert quality["needs_review"] is False


def test_chunk_audit_records_length_and_text_quality_metadata():
    chunks = [
        Chunk(id="short", text="brief", metadata={"source_path": "test.pdf"}),
        Chunk(
            id="bad",
            text="systemoutputshouldbetailoredtodifferenttypesofusers",
            metadata={"source_path": "test.pdf"},
        ),
    ]

    audit = assess_chunk_quality(chunks, target_size=1000)

    assert audit["too_short_count"] == 2
    assert "bad" in audit["flagged_chunk_ids"]
    assert chunks[1].metadata["chunk_quality"]["needs_review"] is True
