# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from evaluation.metrics import answerability_metrics, latency_summary, retrieval_metrics


def test_retrieval_metrics_use_supported_query_denominator_and_dedupe_duplicates() -> None:
    metrics = retrieval_metrics(
        {
            "supported-1": ["p1", "p1", "noise", "p2"],
            "supported-2": ["noise"],
            "unsupported": ["noise"],
            "unlabeled": ["p1"],
        },
        {
            "supported-1": ["p1", "p2"],
            "supported-2": ["p3"],
            "unsupported": [],
        },
    )

    assert metrics["supported_query_count"] == 2
    assert metrics["unsupported_query_count"] == 1
    assert metrics["missing_annotation_query_count"] == 1
    assert metrics["recall_at_k"] == {"1": 0.25, "3": 0.5, "5": 0.5}
    assert metrics["precision_at_k"] == {
        "1": 0.5,
        "3": pytest.approx(1 / 3),
        "5": pytest.approx(0.2),
    }
    assert metrics["mrr"] == 0.5


def test_empty_retrieval_metrics_report_undefined_means() -> None:
    metrics = retrieval_metrics({}, {})

    assert metrics["supported_query_count"] == 0
    assert metrics["recall_at_k"] == {"1": None, "3": None, "5": None}
    assert metrics["mrr"] is None


def test_invalid_k_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        retrieval_metrics({}, {}, k_values=(0,))


def test_answerability_metrics_separate_false_answers_from_abstentions() -> None:
    metrics = answerability_metrics(
        {"supported": True, "unsupported": False, "supported-abstain": True},
        {"supported": True, "unsupported": True, "supported-abstain": False},
    )

    assert metrics["true_answer_count"] == 1
    assert metrics["unsupported_answer_count"] == 1
    assert metrics["supported_abstention_count"] == 1
    assert metrics["unsupported_answer_rate"] == 1.0
    assert metrics["supported_abstention_rate"] == 0.5


def test_latency_summary_handles_empty_small_and_large_samples() -> None:
    assert latency_summary([]) == {"count": 0, "mean": None, "p50": None, "p95": None}
    small = latency_summary([0.2, 0.1, 0.3])
    assert small["mean"] == pytest.approx(0.2)
    assert small["p50"] == 0.2
    assert small["p95"] is None
    large = latency_summary([float(value) for value in range(1, 21)])
    assert large["p95"] == 19


def test_stable_tie_break_uses_chunk_id_in_evaluator() -> None:
    from localrag.chunker import ChunkRecord
    from localrag.retrieval import SearchResult
    from evaluation.run import _stable_result_order

    tied = [
        SearchResult(ChunkRecord("z", "doc.txt", None, "z-id", 0), 0.5),
        SearchResult(ChunkRecord("a", "doc.txt", None, "a-id", 0), 0.5),
    ]
    assert [item.chunk.id for item in _stable_result_order(tied)] == ["a-id", "z-id"]




def test_stable_passage_annotations_cover_both_chunk_boundaries() -> None:
    import json
    from pathlib import Path

    from evaluation.run import DEFAULT_PASSAGES, _chunk_passages
    from localrag.chunker import chunk_documents
    from localrag.loader import load_document

    documents = Path(__file__).parents[1] / "evaluation" / "documents"
    records = [load_document(path)[0] for path in sorted(documents.glob("*.txt"))]
    annotations = json.loads(DEFAULT_PASSAGES.read_text(encoding="utf-8"))["passages"]

    covered_by_config = []
    for size, overlap in ((200, 40), (400, 80)):
        chunks = chunk_documents(records, chunk_size=size, overlap=overlap)
        labels = _chunk_passages(chunks, records, annotations, size, overlap)
        covered_by_config.append({label for labels_for_chunk in labels.values() for label in labels_for_chunk})

    expected = {annotation["id"] for annotation in annotations}
    assert covered_by_config == [expected, expected]
