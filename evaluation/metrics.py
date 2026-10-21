# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Deterministic ranking, answerability, and latency metric helpers."""

from collections.abc import Mapping, Sequence
import math


def retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    relevant: Mapping[str, Sequence[str] | None],
    k_values: Sequence[int] = (1, 3, 5),
) -> dict[str, object]:
    """Compute ranking metrics over answerable queries with gold annotations.

    Recall@k is hits among the first k unique retrieved passage IDs divided by
    the query's unique gold relevant passage count. Precision@k divides by k
    even when fewer than k results exist. MRR uses the first relevant passage.
    Queries labeled with an empty relevant list are counted as unsupported and
    excluded from ranking relevance denominators. Missing annotations are
    reported and excluded as well.
    """
    if any(not isinstance(k, int) or k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")

    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    precisions: dict[int, list[float]] = {k: [] for k in k_values}
    reciprocal_ranks: list[float] = []
    supported_count = 0
    unsupported_count = 0
    missing_annotations = 0

    for query_id, result_ids in rankings.items():
        if query_id not in relevant or relevant[query_id] is None:
            missing_annotations += 1
            continue
        gold = set(relevant[query_id] or ())
        if not gold:
            unsupported_count += 1
            continue
        supported_count += 1
        # Repeated chunks pointing to the same source passage count once.
        ranked: list[str] = []
        seen: set[str] = set()
        for result_id in result_ids:
            if result_id not in seen:
                seen.add(result_id)
                ranked.append(result_id)

        first_rank = next((rank for rank, item in enumerate(ranked, start=1) if item in gold), None)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        for k in k_values:
            hits = sum(item in gold for item in ranked[:k])
            recalls[k].append(hits / len(gold))
            precisions[k].append(hits / k)

    mean = lambda values: sum(values) / len(values) if values else None
    return {
        "supported_query_count": supported_count,
        "unsupported_query_count": unsupported_count,
        "missing_annotation_query_count": missing_annotations,
        "recall_at_k": {str(k): mean(values) for k, values in recalls.items()},
        "precision_at_k": {str(k): mean(values) for k, values in precisions.items()},
        "mrr": mean(reciprocal_ranks),
    }


def answerability_metrics(
    answerable: Mapping[str, bool], predicted_answerable: Mapping[str, bool]
) -> dict[str, int | float]:
    """Compare answerability labels with answer/abstain predictions."""
    query_ids = [query_id for query_id in answerable if query_id in predicted_answerable]
    tp = sum(answerable[qid] and predicted_answerable[qid] for qid in query_ids)
    tn = sum(not answerable[qid] and not predicted_answerable[qid] for qid in query_ids)
    fp = sum(not answerable[qid] and predicted_answerable[qid] for qid in query_ids)
    fn = sum(answerable[qid] and not predicted_answerable[qid] for qid in query_ids)
    unsupported_count = sum(not answerable[qid] for qid in query_ids)
    supported_count = sum(answerable[qid] for qid in query_ids)
    return {
        "evaluated_count": len(query_ids),
        "supported_count": supported_count,
        "unsupported_count": unsupported_count,
        "true_answer_count": tp,
        "correct_abstention_count": tn,
        "unsupported_answer_count": fp,
        "supported_abstention_count": fn,
        "unsupported_answer_rate": fp / unsupported_count if unsupported_count else 0.0,
        "supported_abstention_rate": fn / supported_count if supported_count else 0.0,
    }


def latency_summary(samples_seconds: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize latencies; expose p95 after at least 20 samples."""
    values = sorted(float(value) for value in samples_seconds)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must be finite and non-negative")
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}

    def percentile(percent: float) -> float:
        # Nearest-rank percentile, with one-based rank.
        index = max(0, math.ceil(percent * len(values)) - 1)
        return values[index]

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95) if len(values) >= 20 else None,
    }
