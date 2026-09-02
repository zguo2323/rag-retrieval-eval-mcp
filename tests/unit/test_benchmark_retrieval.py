import json

import pytest

from scripts.benchmark_retrieval import (
    _build_rerank_diagnostics,
    _classify_candidate_failure,
    _load_test_set,
    _parse_fusion_weights,
    _percentile,
    _rank_with_fusion_weight,
    _relevant_ranks,
    _result_score_ids,
    _write_report_json,
    compute_metrics,
)
from src.core.types import RetrievalResult


def test_compute_metrics_with_two_relevant_hits() -> None:
    metrics = compute_metrics(["noise", "a", "b"], ["a", "b"], 5)

    assert metrics["hit_rate@5"] == 1.0
    assert metrics["mrr@5"] == 0.5
    assert metrics["recall@5"] == 1.0
    assert 0.0 < metrics["ndcg@5"] < 1.0


def test_compute_metrics_miss_at_cutoff() -> None:
    metrics = compute_metrics(["noise", "a"], ["a"], 1)

    assert metrics == {
        "hit_rate@1": 0.0,
        "mrr@1": 0.0,
        "recall@1": 0.0,
        "ndcg@1": 0.0,
    }


def test_percentile_interpolates_small_smoke_sample() -> None:
    assert _percentile([10.0, 20.0, 30.0], 0.50) == 20.0
    assert _percentile([10.0, 20.0, 30.0], 0.95) == 29.0


def test_load_test_set_accepts_unanswerable_case(tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "Not in the corpus?",
                        "answerable": False,
                        "expected_chunk_ids": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _load_test_set(path)["test_cases"][0]["answerable"] is False


def test_load_test_set_rejects_unlabelled_answerable_case(tmp_path) -> None:
    path = tmp_path / "golden.json"
    path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "Missing label",
                        "answerable": True,
                        "expected_chunk_ids": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Answerable cases need"):
        _load_test_set(path)


def test_relevant_ranks_and_candidate_failure_classification() -> None:
    dense = _relevant_ranks(["noise", "a"], ["a", "b"])
    sparse = _relevant_ranks(["noise"], ["a", "b"])
    fusion = _relevant_ranks(["noise"], ["a", "b"])

    assert dense == {"a": 2, "b": None}
    assert _classify_candidate_failure(dense, sparse, fusion) == "fusion_drop"


def test_result_score_ids_prefer_stable_metadata_key() -> None:
    results = [
        RetrievalResult(
            chunk_id="abcd1234_0003_newhash",
            score=0.1,
            text="",
            metadata={"stable_chunk_key": "abcd1234_0003"},
        ),
        RetrievalResult(
            chunk_id="wxyz9999_0010_oldhash",
            score=0.1,
            text="",
            metadata={},
        ),
    ]

    assert _result_score_ids(results, stable_chunk_index=True) == [
        "abcd1234_0003",
        "wxyz9999_0010",
    ]
    assert _result_score_ids(results, stable_chunk_index=False) == [
        "abcd1234_0003_newhash",
        "wxyz9999_0010_oldhash",
    ]


def test_candidate_failure_detects_dual_and_partial_recall() -> None:
    missed = {"a": None, "b": None}
    partial = {"a": 3, "b": None}

    assert _classify_candidate_failure(missed, missed, missed) == "dual_retrieval_miss"
    assert _classify_candidate_failure(partial, missed, partial) == "partial_fusion_recall"


def test_build_rerank_diagnostics_exposes_ce_and_combined_ranks() -> None:
    results = [
        RetrievalResult(
            chunk_id="noise",
            score=0.1,
            text="noise",
            metadata={"cross_encoder_rank": 1, "fusion_rank": 2},
        ),
        RetrievalResult(
            chunk_id="gold",
            score=0.09,
            text="gold",
            metadata={
                "cross_encoder_rank": 3,
                "fusion_rank": 1,
                "rerank_score": -0.5,
                "input_token_count": 520,
                "model_max_length": 512,
                "input_truncated": True,
            },
        ),
    ]

    diagnostics = _build_rerank_diagnostics(results, ["gold", "missing"])

    assert diagnostics["relevant_chunks"][0]["combined_rank"] == 2
    assert diagnostics["relevant_chunks"][0]["cross_encoder_rank"] == 3
    assert diagnostics["relevant_chunks"][0]["input_truncated"] is True
    assert diagnostics["relevant_chunks"][1]["not_in_rerank_pool"] is True


def test_parse_fusion_weights_validates_and_deduplicates() -> None:
    assert _parse_fusion_weights("0.6,0.5,0.6") == [0.6, 0.5]
    with pytest.raises(ValueError, match="between 0 and 1"):
        _parse_fusion_weights("1.1")


def test_rank_with_fusion_weight_reuses_recorded_ranks() -> None:
    fusion_favorite = RetrievalResult(
        chunk_id="fusion",
        score=0.0,
        text="",
        metadata={"fusion_rank": 1, "cross_encoder_rank": 10},
    )
    ce_favorite = RetrievalResult(
        chunk_id="ce",
        score=0.0,
        text="",
        metadata={"fusion_rank": 10, "cross_encoder_rank": 1},
    )

    assert _rank_with_fusion_weight([fusion_favorite, ce_favorite], 0.6, 60)[0] == "fusion"
    assert _rank_with_fusion_weight([fusion_favorite, ce_favorite], 0.4, 60)[0] == "ce"


def test_write_report_json_creates_parent_directory(tmp_path) -> None:
    path = tmp_path / "reports" / "dev.json"

    written = _write_report_json({"metric": 0.714}, path)

    assert written == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"metric": 0.714}
