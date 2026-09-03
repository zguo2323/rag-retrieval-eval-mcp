#!/usr/bin/env python
"""Run a stage-by-stage retrieval benchmark against a golden set.

The benchmark keeps Test frozen by default, reports Dense/Sparse/Fusion/Rerank
quality at a shared cutoff, and exposes opt-in diagnostics for Dev-only strategy
work such as rerank weight sweeps, candidate failure analysis, and ingestion A/B
checks with stable chunk-index matching.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.query_engine.dense_retriever import create_dense_retriever
from src.core.query_engine.hybrid_search import HybridSearchResult, create_hybrid_search
from src.core.query_engine.query_processor import QueryProcessor, QueryProcessorConfig
from src.core.query_engine.llm_query_expander import LLMQueryExpander, LLMQueryExpanderConfig
from src.core.query_engine.reranker import create_core_reranker
from src.core.query_engine.sparse_retriever import create_sparse_retriever
from src.core.settings import load_settings
from src.core.trace import TraceContext
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.llm.llm_factory import LLMFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory

STAGES = ("dense", "sparse", "fusion", "rerank")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retrieval stages.")
    parser.add_argument(
        "--test-set",
        default="data/benchmark/golden_nist_ai_40.json",
        help="Golden-set JSON path.",
    )
    parser.add_argument("--collection", default=None)
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--k", type=int, default=5, help="Common metric cutoff.")
    parser.add_argument(
        "--split",
        choices=("all", "dev", "test"),
        default="dev",
        help="Evaluate dev by default, or explicitly select test/all.",
    )
    parser.add_argument(
        "--case-ids",
        default=None,
        help="Comma-separated case IDs to run within the selected split.",
    )
    parser.add_argument(
        "--fusion-weight-sweep",
        default=None,
        help=(
            "Comma-separated Fusion rank weights to evaluate from the same "
            "retrieval and Cross-Encoder run, for example 0.6,0.5,0.4."
        ),
    )
    parser.add_argument(
        "--query-expansion",
        action="store_true",
        help="Enable deterministic synonym expansion for this benchmark run.",
    )
    parser.add_argument(
        "--llm-query-expansion",
        action="store_true",
        help="Enable the configured failure-safe LLM retrieval rewrite.",
    )
    parser.add_argument("--llm-query-hard-fallback", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Write the full JSON report to this path for later comparison.",
    )
    parser.add_argument(
        "--match-stable-chunk-index",
        action="store_true",
        help=(
            "For ingestion A/B only: score chunk IDs by source-path hash prefix "
            "and chunk_index, ignoring the content-hash suffix."
        ),
    )
    return parser.parse_args()


def _load_test_set(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Golden set must contain a non-empty test_cases list")
    for case in cases:
        if not case.get("query"):
            raise ValueError("Every case needs a query")
        if not isinstance(case.get("answerable", True), bool):
            raise ValueError("Every case needs a boolean answerable field")
        expected_ids = case.get("expected_chunk_ids")
        if not isinstance(expected_ids, list):
            raise ValueError("Every case needs an expected_chunk_ids list")
        if case.get("answerable", True) and not expected_ids:
            raise ValueError("Answerable cases need expected_chunk_ids")
        if not case.get("answerable", True) and expected_ids:
            raise ValueError("Unanswerable cases cannot have expected_chunk_ids")
    return payload


def _write_report_json(report: dict[str, Any], path: str | Path) -> Path:
    """Persist a complete benchmark report without relying on terminal output."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _ids(results: Iterable[Any]) -> list[str]:
    return [str(result.chunk_id) for result in results]


def _stable_chunk_position_id(chunk_id: str) -> str:
    parts = chunk_id.split("_")
    if len(parts) >= 3 and parts[1].isdigit():
        return f"{parts[0]}_{parts[1]}"
    return chunk_id


def _result_score_id(result: Any, stable_chunk_index: bool) -> str:
    if not stable_chunk_index:
        return str(result.chunk_id)
    metadata = getattr(result, "metadata", {}) or {}
    stable_key = metadata.get("stable_chunk_key")
    if stable_key:
        return str(stable_key)
    return _stable_chunk_position_id(str(result.chunk_id))


def _result_score_ids(results: Iterable[Any], stable_chunk_index: bool) -> list[str]:
    return [_result_score_id(result, stable_chunk_index) for result in results]


def _score_ids(ids: Sequence[str], stable_chunk_index: bool) -> list[str]:
    if not stable_chunk_index:
        return list(ids)
    return [_stable_chunk_position_id(chunk_id) for chunk_id in ids]


def _relevant_ranks(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> dict[str, int | None]:
    ranked_ids = list(retrieved_ids)
    rank_by_id = {chunk_id: rank for rank, chunk_id in enumerate(ranked_ids, 1)}
    return {
        chunk_id: rank_by_id.get(chunk_id)
        for chunk_id in relevant_ids
    }


def _classify_candidate_failure(
    dense_ranks: dict[str, int | None],
    sparse_ranks: dict[str, int | None],
    fusion_ranks: dict[str, int | None],
) -> str:
    route_hit = any(rank is not None for rank in dense_ranks.values()) or any(
        rank is not None for rank in sparse_ranks.values()
    )
    fusion_hit_count = sum(rank is not None for rank in fusion_ranks.values())
    if not route_hit:
        return "dual_retrieval_miss"
    if fusion_hit_count == 0:
        return "fusion_drop"
    if fusion_hit_count < len(fusion_ranks):
        return "partial_fusion_recall"
    return "fusion_recalled"


def _build_rerank_diagnostics(
    ranked_results: Sequence[Any],
    relevant_ids: Sequence[str],
    stable_chunk_index: bool = False,
) -> dict[str, Any]:
    """Expose full rerank positions and CE inputs without affecting metrics."""
    relevant = set(_score_ids(relevant_ids, stable_chunk_index))
    by_id = {
        _result_score_id(result, stable_chunk_index): result
        for result in ranked_results
    }
    combined_rank = {
        result.chunk_id: rank for rank, result in enumerate(ranked_results, 1)
    }

    def describe(result: Any) -> dict[str, Any]:
        metadata = result.metadata
        return {
            "chunk_id": result.chunk_id,
            "fusion_rank": metadata.get("fusion_rank"),
            "cross_encoder_rank": metadata.get("cross_encoder_rank"),
            "combined_rank": combined_rank[result.chunk_id],
            "cross_encoder_score": metadata.get("rerank_score"),
            "input_token_count": metadata.get("input_token_count"),
            "model_max_length": metadata.get("model_max_length"),
            "input_truncated": metadata.get("input_truncated"),
        }

    relevant_diagnostics = []
    for chunk_id in relevant_ids:
        result = by_id.get(_score_ids([chunk_id], stable_chunk_index)[0])
        relevant_diagnostics.append(
            describe(result)
            if result is not None
            else {"chunk_id": chunk_id, "not_in_rerank_pool": True}
        )
    ce_top5 = sorted(
        ranked_results,
        key=lambda result: result.metadata.get("cross_encoder_rank", math.inf),
    )[:5]
    return {
        "relevant_chunks": relevant_diagnostics,
        "cross_encoder_top5": [
            {
                **describe(result),
                "is_relevant": (
                    _result_score_id(result, stable_chunk_index) in relevant
                ),
            }
            for result in ce_top5
        ],
    }


def _parse_fusion_weights(value: str | None) -> list[float]:
    if not value:
        return []
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("Fusion sweep weights must be between 0 and 1")
    return list(dict.fromkeys(weights))


def _rank_with_fusion_weight(
    ranked_results: Sequence[Any], weight: float, constant: int
) -> list[str]:
    """Recompute weighted RRF from recorded Fusion and CE ranks."""
    if any(
        result.metadata.get("fusion_rank") is None
        or result.metadata.get("cross_encoder_rank") is None
        for result in ranked_results
    ):
        raise ValueError("Weight sweep requires rank_fusion diagnostics")
    reordered = sorted(
        ranked_results,
        key=lambda result: (
            weight / (constant + result.metadata["fusion_rank"])
            + (1.0 - weight)
            / (constant + result.metadata["cross_encoder_rank"])
        ),
        reverse=True,
    )
    return _ids(reordered)


def compute_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    k: int,
) -> dict[str, float]:
    """Compute binary-relevance IR metrics at a shared cutoff."""
    ranked = list(retrieved_ids[:k])
    relevant = set(relevant_ids)
    hits = [1 if chunk_id in relevant else 0 for chunk_id in ranked]
    hit_count = len(set(ranked) & relevant)

    first_rank = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, 1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return {
        f"hit_rate@{k}": 1.0 if hit_count else 0.0,
        f"mrr@{k}": 1.0 / first_rank if first_rank else 0.0,
        f"recall@{k}": hit_count / len(relevant) if relevant else 0.0,
        f"ndcg@{k}": dcg / idcg if idcg else 0.0,
    }


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _build_components(
    settings: Any, collection: str, query_expansion: bool = False,
    llm_query_expansion: bool = False,
    llm_query_hard_fallback: bool = False,
) -> tuple[Any, Any]:
    vector_store = VectorStoreFactory.create(settings, collection_name=collection)
    embedding_client = EmbeddingFactory.create(settings)
    dense = create_dense_retriever(settings, embedding_client, vector_store)
    bm25 = BM25Indexer(index_dir=f"data/db/bm25/{collection}")
    sparse = create_sparse_retriever(settings, bm25, vector_store)
    sparse.default_collection = collection
    llm_expander = None
    if llm_query_expansion:
        config = getattr(settings.retrieval, "llm_query_expansion", {}) or {}
        llm_expander = LLMQueryExpander(
            LLMFactory.create(settings),
            LLMQueryExpanderConfig(
                max_tokens=int(config.get("max_tokens", 96)),
                timeout_seconds=float(config.get("timeout_seconds", 8.0)),
            ),
        )
    hybrid = create_hybrid_search(
        settings=settings,
        query_processor=QueryProcessor(
            QueryProcessorConfig(enable_query_expansion=query_expansion)
        ),
        dense_retriever=dense,
        sparse_retriever=sparse,
        query_expander=llm_expander,
    )
    hybrid.config.llm_query_hard_fallback = llm_query_hard_fallback or (
        config.get("mode") == "hard_fallback" if llm_query_expansion else False
    )
    return hybrid, create_core_reranker(settings)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_test_set(args.test_set)
    selected_cases = [
        case
        for case in payload["test_cases"]
        if args.split == "all" or case.get("split") == args.split
    ]
    if args.case_ids:
        requested_ids = {
            case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()
        }
        selected_cases = [
            case for case in selected_cases if case.get("id") in requested_ids
        ]
        found_ids = {case.get("id") for case in selected_cases}
        missing_ids = sorted(requested_ids - found_ids)
        if missing_ids:
            raise ValueError(
                f"Case IDs not found in split {args.split}: {', '.join(missing_ids)}"
            )
    if not selected_cases:
        raise ValueError(f"No cases found for split: {args.split}")
    collection = args.collection or payload.get("collection")
    if not collection:
        raise ValueError("Collection is required in the golden set or --collection")
    if args.k < 1:
        raise ValueError("--k must be at least 1")

    settings = load_settings(args.config)
    hybrid, reranker = _build_components(
        settings, collection, query_expansion=args.query_expansion,
        llm_query_expansion=args.llm_query_expansion,
        llm_query_hard_fallback=args.llm_query_hard_fallback,
    )
    fusion_weights = _parse_fusion_weights(args.fusion_weight_sweep)
    if fusion_weights and reranker.config.strategy != "rank_fusion":
        raise ValueError("--fusion-weight-sweep requires rerank.strategy=rank_fusion")
    per_query: list[dict[str, Any]] = []
    latency: dict[str, list[float]] = {stage: [] for stage in STAGES}
    query_expansion_latency: list[float] = []
    end_to_end_search_latency: list[float] = []
    fallback_count = 0
    llm_query_expansion_fallback_count = 0

    for case in selected_cases:
        trace = TraceContext(trace_type="query")
        search_started = time.monotonic()
        detailed = hybrid.search(
            query=case["query"], trace=trace, return_details=True
        )
        search_ms = (time.monotonic() - search_started) * 1000
        end_to_end_search_latency.append(search_ms)
        if not isinstance(detailed, HybridSearchResult):
            raise RuntimeError("Hybrid search did not return detailed results")
        if detailed.query_expansion is not None:
            query_expansion_latency.append(detailed.query_expansion.elapsed_ms)
            llm_query_expansion_fallback_count += int(
                detailed.query_expansion.used_fallback
            )

        stage_results = {
            "dense": detailed.dense_results or [],
            "sparse": detailed.sparse_results or [],
            "fusion": detailed.results,
        }
        fallback_count += int(detailed.used_fallback)
        for name, trace_name in (
            ("dense", "dense_retrieval"),
            ("sparse", "sparse_retrieval"),
            ("fusion", "fusion"),
        ):
            try:
                latency[name].append(trace.elapsed_ms(trace_name))
            except KeyError:
                latency[name].append(search_ms if name == "fusion" else 0.0)

        rerank_started = time.monotonic()
        reranked = reranker.rerank(
            query=detailed.rerank_query or case["query"],
            results=detailed.results,
            top_k=args.k,
            trace=trace,
            include_diagnostics=True,
        )
        latency["rerank"].append((time.monotonic() - rerank_started) * 1000)
        fallback_count += int(reranked.used_fallback)
        stage_results["rerank"] = reranked.results
        full_rerank_results = reranked.ranked_results or reranked.results
        weight_sweep_ids = {
            str(weight): _rank_with_fusion_weight(
                full_rerank_results, weight, reranker.config.rank_fusion_k
            )
            for weight in fusion_weights
        }

        answerable = case.get("answerable", True)
        metrics = (
            {
                stage: compute_metrics(
                    _result_score_ids(results, args.match_stable_chunk_index),
                    _score_ids(
                        case["expected_chunk_ids"],
                        args.match_stable_chunk_index,
                    ),
                    args.k,
                )
                for stage, results in stage_results.items()
            }
            if answerable
            else {stage: None for stage in STAGES}
        )
        relevant_ranks = {
            stage: _relevant_ranks(
                _result_score_ids(results, args.match_stable_chunk_index),
                _score_ids(case["expected_chunk_ids"], args.match_stable_chunk_index),
            )
            for stage, results in stage_results.items()
        }
        candidate_failure = (
            _classify_candidate_failure(
                relevant_ranks["dense"],
                relevant_ranks["sparse"],
                relevant_ranks["fusion"],
            )
            if answerable
            else None
        )
        scored_relevant = set(
            _score_ids(case["expected_chunk_ids"], args.match_stable_chunk_index)
        )
        fusion_candidate_hits = scored_relevant & set(
            _result_score_ids(detailed.results, args.match_stable_chunk_index)
        )
        rerank_hits = scored_relevant & set(
            _result_score_ids(reranked.results[: args.k], args.match_stable_chunk_index)
        )
        retained_hits = fusion_candidate_hits & rerank_hits
        rerank_mrr_delta = None
        if answerable:
            rerank_mrr_delta = (
                metrics["rerank"][f"mrr@{args.k}"]
                - metrics["fusion"][f"mrr@{args.k}"]
            )
        per_query.append(
            {
                "id": case.get("id"),
                "query": case["query"],
                "category": case.get("category"),
                "split": case.get("split"),
                "answerable": answerable,
                "relevant_ids": case["expected_chunk_ids"],
                "relevant_ranks": relevant_ranks,
                "candidate_failure": candidate_failure,
                "query_expansion": (
                    {
                        "rewrite": detailed.query_expansion.rewrite,
                        "elapsed_ms": detailed.query_expansion.elapsed_ms,
                        "used_fallback": detailed.query_expansion.used_fallback,
                        "fallback_reason": detailed.query_expansion.fallback_reason,
                    }
                    if detailed.query_expansion is not None
                    else None
                ),
                "rerank_query": detailed.rerank_query,
                "stages": {
                    stage: {
                        "retrieved_ids": _ids(results)[: args.k],
                        "metrics": metrics[stage],
                    }
                    for stage, results in stage_results.items()
                },
                "rerank_mrr_delta": rerank_mrr_delta,
                "fusion_candidate_relevant_count": (
                    len(fusion_candidate_hits) if answerable else None
                ),
                "rerank_retained_relevant_count": (
                    len(retained_hits) if answerable else None
                ),
                "rerank_candidate_retention": (
                    len(retained_hits) / len(fusion_candidate_hits)
                    if answerable and fusion_candidate_hits
                    else (1.0 if answerable else None)
                ),
                "rerank_lost_relevant_ids": (
                    sorted(fusion_candidate_hits - rerank_hits)
                    if answerable
                    else None
                ),
                "rerank_fallback": reranked.used_fallback,
                "rerank_diagnostics": _build_rerank_diagnostics(
                    full_rerank_results,
                    case["expected_chunk_ids"],
                    stable_chunk_index=args.match_stable_chunk_index,
                ),
                "fusion_weight_sweep": {
                    weight: {
                        "retrieved_ids": ids[: args.k],
                        "metrics": (
                            compute_metrics(
                                _score_ids(ids, args.match_stable_chunk_index),
                                _score_ids(
                                    case["expected_chunk_ids"],
                                    args.match_stable_chunk_index,
                                ),
                                args.k,
                            )
                            if answerable
                            else None
                        ),
                    }
                    for weight, ids in weight_sweep_ids.items()
                },
            }
        )

    answerable_rows = [row for row in per_query if row["answerable"]]
    unanswerable_rows = [row for row in per_query if not row["answerable"]]
    if not answerable_rows:
        raise ValueError("Selected split has no answerable cases to score")
    aggregate: dict[str, Any] = {}
    for stage in STAGES:
        aggregate[stage] = {
            metric: statistics.mean(
                row["stages"][stage]["metrics"][metric]
                for row in answerable_rows
            )
            for metric in answerable_rows[0]["stages"][stage]["metrics"]
        }
        aggregate[stage]["p50_ms"] = _percentile(latency[stage], 0.50)
        aggregate[stage]["p95_ms"] = _percentile(latency[stage], 0.95)

    deltas = [row["rerank_mrr_delta"] for row in answerable_rows]
    lost_opportunities = [
        row for row in answerable_rows if row["rerank_lost_relevant_ids"]
    ]
    retention_eligible_rows = [
        row
        for row in answerable_rows
        if row["fusion_candidate_relevant_count"] > 0
    ]
    weight_sweep = {}
    for weight in map(str, fusion_weights):
        weight_metrics = [
            row["fusion_weight_sweep"][weight]["metrics"]
            for row in answerable_rows
        ]
        mrr_deltas = [
            metrics[f"mrr@{args.k}"]
            - row["stages"]["fusion"]["metrics"][f"mrr@{args.k}"]
            for row, metrics in zip(answerable_rows, weight_metrics)
        ]
        weight_sweep[weight] = {
            **{
                metric: statistics.mean(row[metric] for row in weight_metrics)
                for metric in weight_metrics[0]
            },
            "mean_mrr_delta_vs_fusion": statistics.mean(mrr_deltas),
            "improved_queries": sum(delta > 0 for delta in mrr_deltas),
            "unchanged_queries": sum(delta == 0 for delta in mrr_deltas),
            "degraded_queries": sum(delta < 0 for delta in mrr_deltas),
        }
    return {
        "benchmark": payload.get("name", Path(args.test_set).stem),
        "collection": collection,
        "split": args.split,
        "query_count": len(per_query),
        "answerable_query_count": len(answerable_rows),
        "unanswerable_query_count": len(unanswerable_rows),
        "metric_cutoff": args.k,
        "query_expansion": args.query_expansion,
        "llm_query_expansion": args.llm_query_expansion,
        "llm_query_hard_fallback": hybrid.config.llm_query_hard_fallback,
        "candidate_top_k": {
            "dense": hybrid.config.dense_top_k,
            "sparse": hybrid.config.sparse_top_k,
            "fusion": hybrid.config.fusion_top_k,
            "rerank": args.k,
        },
        "aggregate": aggregate,
        "latency": {
            "end_to_end_search_p50_ms": _percentile(end_to_end_search_latency, 0.50),
            "end_to_end_search_p95_ms": _percentile(end_to_end_search_latency, 0.95),
            "query_expansion_p50_ms": _percentile(query_expansion_latency, 0.50),
            "query_expansion_p95_ms": _percentile(query_expansion_latency, 0.95),
        },
        "rerank_analysis": {
            "mean_mrr_delta": statistics.mean(deltas),
            "improved_queries": sum(delta > 0 for delta in deltas),
            "unchanged_queries": sum(delta == 0 for delta in deltas),
            "degraded_queries": sum(delta < 0 for delta in deltas),
            "candidate_retention_mean": statistics.mean(
                row["rerank_candidate_retention"]
                for row in retention_eligible_rows
            ) if retention_eligible_rows else None,
            "candidate_retention_eligible_queries": len(retention_eligible_rows),
            "candidate_opportunity_loss_queries": len(lost_opportunities),
        },
        "fusion_weight_sweep": weight_sweep,
        "candidate_failure_counts": {
            classification: sum(
                row["candidate_failure"] == classification
                for row in answerable_rows
            )
            for classification in (
                "dual_retrieval_miss",
                "fusion_drop",
                "partial_fusion_recall",
                "fusion_recalled",
            )
        },
        "fallback_rate": fallback_count / (len(per_query) * 2),
        "llm_query_expansion_fallback_rate": (
            llm_query_expansion_fallback_count / len(per_query)
            if args.llm_query_expansion
            else 0.0
        ),
        "unanswerable_diagnostics": [
            {
                "id": row["id"],
                "query": row["query"],
                "returned_counts": {
                    stage: len(row["stages"][stage]["retrieved_ids"])
                    for stage in STAGES
                },
                "top_ids": {
                    stage: row["stages"][stage]["retrieved_ids"]
                    for stage in STAGES
                },
                "note": "Diagnostic only: retrieval has no abstention decision yet.",
            }
            for row in unanswerable_rows
        ],
        "queries": per_query,
        "latency_note": (
            "P50/P95 use one run per selected query; add warmup and repeated runs "
            "before treating them as reportable latency benchmarks."
        ),
    }


def _print_report(report: dict[str, Any]) -> None:
    k = report["metric_cutoff"]
    print(
        f"Benchmark: {report['benchmark']} ({report['query_count']} queries: "
        f"{report['answerable_query_count']} scored, "
        f"{report['unanswerable_query_count']} diagnostic)"
    )
    print(
        f"Collection: {report['collection']} | split: {report['split']} "
        f"| metric cutoff: @{k}"
    )
    print()
    header = f"{'Stage':<9} {'Hit@'+str(k):>8} {'MRR@'+str(k):>8} {'Recall@'+str(k):>10} {'NDCG@'+str(k):>9} {'P50 ms':>10} {'P95 ms':>10}"
    print(header)
    print("-" * len(header))
    for stage in STAGES:
        row = report["aggregate"][stage]
        print(
            f"{stage:<9} {row[f'hit_rate@{k}']:>8.3f} {row[f'mrr@{k}']:>8.3f} "
            f"{row[f'recall@{k}']:>10.3f} {row[f'ndcg@{k}']:>9.3f} "
            f"{row['p50_ms']:>10.1f} {row['p95_ms']:>10.1f}"
        )
    analysis = report["rerank_analysis"]
    retention_mean = analysis["candidate_retention_mean"]
    retention_text = f"{retention_mean:.3f}" if retention_mean is not None else "n/a"
    print()
    print(
        "Rerank vs Fusion MRR: "
        f"mean delta={analysis['mean_mrr_delta']:+.3f}, "
        f"improved={analysis['improved_queries']}, "
        f"unchanged={analysis['unchanged_queries']}, "
        f"degraded={analysis['degraded_queries']}"
    )
    print(
        "Rerank candidate retention: "
        f"mean={retention_text}, "
        f"eligible={analysis['candidate_retention_eligible_queries']}, "
        f"opportunity-loss queries={analysis['candidate_opportunity_loss_queries']}"
    )
    print(f"Fallback rate: {report['fallback_rate']:.1%}")
    print(f"Candidate analysis: {report['candidate_failure_counts']}")
    if report["fusion_weight_sweep"]:
        print("Fusion weight sweep (same candidates and CE scores):")
        for weight, row in report["fusion_weight_sweep"].items():
            print(
                f"  weight={weight}: Hit@{k}={row[f'hit_rate@{k}']:.3f}, "
                f"MRR@{k}={row[f'mrr@{k}']:.3f}, "
                f"Recall@{k}={row[f'recall@{k}']:.3f}, "
                f"NDCG@{k}={row[f'ndcg@{k}']:.3f}, "
                f"delta={row['mean_mrr_delta_vs_fusion']:+.3f}, "
                f"improved/degraded={row['improved_queries']}/"
                f"{row['degraded_queries']}"
            )
    print(f"Note: {report['latency_note']}")
    print()
    for row in report["queries"]:
        if not row["answerable"]:
            print(f"[{row['id']}] unanswerable diagnostic (not quality-scored)")
            for stage in STAGES:
                ids = row["stages"][stage]["retrieved_ids"]
                print(f"  {stage:<7} ids={','.join(ids)}")
            continue
        print(
            f"[{row['id']}] rerank MRR delta={row['rerank_mrr_delta']:+.3f}, "
            f"candidate retention={row['rerank_candidate_retention']:.3f}, "
            f"lost={','.join(row['rerank_lost_relevant_ids']) or '-'}"
        )
        print(
            f"  candidate={row['candidate_failure']} "
            f"ranks={row['relevant_ranks']}"
        )
        print(
            "  rerank diagnostics="
            f"{row['rerank_diagnostics']['relevant_chunks']}"
        )
        print(
            "  CE top5="
            f"{row['rerank_diagnostics']['cross_encoder_top5']}"
        )
        for stage in STAGES:
            metrics = row["stages"][stage]["metrics"]
            ids = row["stages"][stage]["retrieved_ids"]
            print(
                f"  {stage:<7} hit={metrics[f'hit_rate@{k}']:.0f} "
                f"mrr={metrics[f'mrr@{k}']:.3f} recall={metrics[f'recall@{k}']:.3f} "
                f"ids={','.join(ids)}"
            )


def main() -> int:
    args = parse_args()
    try:
        report = run_benchmark(args)
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    if args.output_json:
        output_path = _write_report_json(report, args.output_json)
        print(f"Benchmark JSON written to: {output_path}", file=sys.stderr)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
