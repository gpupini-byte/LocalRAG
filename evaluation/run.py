# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Reproducible synthetic-fixture retrieval and optional answer evaluation."""

import argparse
import csv
from importlib.metadata import PackageNotFoundError, version
import json
import re
import sys
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from evaluation.metrics import answerability_metrics, latency_summary, retrieval_metrics
from localrag.chunker import ChunkRecord, chunk_documents
from localrag.generator import OllamaChatClient
from localrag.loader import DocumentRecord, load_document
from localrag.prompting import build_grounded_messages, make_evidence
from localrag.retrieval import FaissVectorStore, SearchResult


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "dataset.json"
DEFAULT_PASSAGES = ROOT / "passages.json"
DEFAULT_DOCUMENTS = ROOT / "documents"
_WORDS = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
_CITATIONS = re.compile(r"\[(S\d+)\]")


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORDS.finditer(text)]


def _stable_result_order(results: list[SearchResult]) -> list[SearchResult]:
    """Break equal-score retrieval ties deterministically by stable chunk ID."""
    return sorted(results, key=lambda result: (-result.score, result.chunk.id))


def _passage_spans(
    records: list[DocumentRecord], passage_annotations: list[dict[str, str]]
) -> dict[str, list[tuple[str, int, int]]]:
    """Resolve stable passage anchors to whitespace-word offsets per record."""
    annotations_by_document: dict[str, list[dict[str, str]]] = {}
    for annotation in passage_annotations:
        annotations_by_document.setdefault(annotation["document_id"], []).append(annotation)

    result: dict[str, list[tuple[str, int, int]]] = {}
    for record in records:
        document_id = Path(record.source).stem
        annotations = annotations_by_document.get(document_id, [])
        source_tokens = record.text.split()
        normalized = ["".join(_tokens(token)) for token in source_tokens]
        for annotation in annotations:
            anchor_tokens = _tokens(annotation["anchor"])
            anchor_norm = ["".join(_tokens(token)) for token in anchor_tokens]
            start = next(
                (
                    index
                    for index in range(len(normalized) - len(anchor_norm) + 1)
                    if normalized[index : index + len(anchor_norm)] == anchor_norm
                ),
                None,
            )
            if start is None:
                raise ValueError(
                    f"Passage anchor for '{annotation['id']}' was not found in {record.source}"
                )
            result.setdefault(record.source, []).append(
                (annotation["id"], start, start + len(anchor_norm))
            )
    return result


def _chunk_passages(
    chunks: list[ChunkRecord],
    records: list[DocumentRecord],
    passage_annotations: list[dict[str, str]],
    chunk_size: int,
    overlap: int,
) -> dict[str, list[str]]:
    spans = _passage_spans(records, passage_annotations)
    record_word_counts = {record.source: len(record.text.split()) for record in records}
    step = chunk_size - overlap
    labels: dict[str, list[str]] = {}
    for chunk in chunks:
        start = chunk.position * step
        end = min(start + chunk_size, record_word_counts[chunk.source])
        labels[chunk.id] = [
            passage_id
            for passage_id, passage_start, passage_end in spans.get(chunk.source, [])
            if start < passage_end and passage_start < end
        ]
    return labels


def evaluate(
    dataset_path: Path,
    passages_path: Path,
    documents_dir: Path,
    embedding_model: str,
    top_k: int,
    minimum_score: float | None,
    with_generation: bool,
    ollama_model: str,
    api_url: str,
    timeout: float,
) -> dict[str, object]:
    from localrag.embeddings import SentenceTransformerEmbedder

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    passage_data = json.loads(passages_path.read_text(encoding="utf-8"))
    questions = dataset["questions"]
    passage_annotations = passage_data["passages"]
    for question in questions:
        expected = question.get("expected_relevant_passage_ids")
        if expected is None:
            raise ValueError(f"Question {question.get('id')} is missing relevance annotations")
        if bool(expected) != bool(question.get("answerable")):
            raise ValueError(
                f"Question {question['id']} answerability disagrees with passage labels"
            )

    load_start = perf_counter()
    paths = sorted(
        path
        for path in documents_dir.glob("*.txt")
        if path.is_file()
    )
    if not paths:
        raise ValueError(f"No .txt benchmark documents found in '{documents_dir}'")
    records: list[DocumentRecord] = []
    for path in paths:
        records.extend(load_document(path))
    document_load_seconds = perf_counter() - load_start

    model_start = perf_counter()
    embedder = SentenceTransformerEmbedder(embedding_model)
    model_initialization_seconds = perf_counter() - model_start
    generator = (
        OllamaChatClient(model=ollama_model, api_url=api_url, timeout=timeout)
        if with_generation
        else None
    )

    configs = ((200, 40), (400, 80))
    config_results: list[dict[str, object]] = []
    review_rows: list[dict[str, object]] = []
    for chunk_size, overlap in configs:
        chunk_start = perf_counter()
        chunks = chunk_documents(records, chunk_size=chunk_size, overlap=overlap)
        passage_by_chunk = _chunk_passages(
            chunks, records, passage_annotations, chunk_size, overlap
        )
        chunking_seconds = perf_counter() - chunk_start

        embedding_start = perf_counter()
        embedded = embedder.embed(chunks)
        document_embedding_seconds = perf_counter() - embedding_start
        store = FaissVectorStore(
            dimension=embedder.dimension, model_name=embedding_model
        )
        store.add(embedded)

        rankings: dict[str, list[str]] = {}
        relevant: dict[str, list[str]] = {}
        answerable: dict[str, bool] = {}
        predicted_answerable: dict[str, bool] = {}
        rows: list[dict[str, object]] = []
        query_embedding_samples: list[float] = []
        retrieval_samples: list[float] = []
        generation_samples: list[float] = []
        total_query_samples: list[float] = []

        for question in questions:
            query_id = question["id"]
            expected_passages = list(question["expected_relevant_passage_ids"])
            relevant[query_id] = expected_passages
            answerable[query_id] = bool(question["answerable"])

            query_chunk = ChunkRecord(
                text=question["question"], source="<evaluation-query>", page=None,
                id=query_id, position=0
            )
            query_embedding_start = perf_counter()
            query_vector = embedder.embed([query_chunk]).embeddings[0]
            query_embedding_seconds = perf_counter() - query_embedding_start
            query_embedding_samples.append(query_embedding_seconds)

            retrieval_start = perf_counter()
            results = _stable_result_order(store.search(query_vector, top_k=top_k))
            if minimum_score is not None:
                results = [result for result in results if result.score >= minimum_score]
            retrieval_seconds = perf_counter() - retrieval_start
            retrieval_samples.append(retrieval_seconds)

            ranking: list[str] = []
            for result in results:
                passage_ids = passage_by_chunk[result.chunk.id]
                ranking.extend(passage_ids or [f"chunk:{result.chunk.id}"])
            rankings[query_id] = ranking

            generated_answer = ""
            citations: list[str] = []
            status = "retrieved" if results else "insufficient_evidence"
            query_total = query_embedding_seconds + retrieval_seconds
            if with_generation and results:
                evidence = make_evidence(results)
                messages = build_grounded_messages(question["question"], evidence)
                generation_start = perf_counter()
                generated_answer = generator.generate(messages)  # type: ignore[union-attr]
                generation_seconds = perf_counter() - generation_start
                generation_samples.append(generation_seconds)
                query_total += generation_seconds
                citations = list(dict.fromkeys(_CITATIONS.findall(generated_answer)))
                allowed = {item.source_id for item in evidence}
                if any(citation not in allowed for citation in citations):
                    status = "invalid_citations"
                elif not citations and "insufficient evidence" not in generated_answer.casefold():
                    status = "missing_citations"
                elif "insufficient evidence" in generated_answer.casefold():
                    status = "insufficient_evidence"
                else:
                    status = "ok"
            total_query_samples.append(query_total)
            predicted_answerable[query_id] = status == "ok" if with_generation else bool(results)
            row = {
                "question_id": query_id,
                "question": question["question"],
                "answerable": question["answerable"],
                "expected_passage_ids": expected_passages,
                "retrieved_passage_ids": ranking,
                "retrieved_chunk_ids": [result.chunk.id for result in results],
                "scores": [result.score for result in results],
                "status": status,
                "answer": generated_answer,
                "citations": citations,
                "citation_labels_valid": (
                    all(citation in allowed for citation in citations)
                    if with_generation and citations
                    else None
                ),
                "query_embedding_seconds": query_embedding_seconds,
                "retrieval_seconds": retrieval_seconds,
                "generation_seconds": generation_samples[-1] if with_generation and results else None,
                "total_query_seconds": query_total,
            }
            rows.append(row)
            review_rows.append(
                {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "question_id": query_id,
                    "question": question["question"],
                    "answer": generated_answer,
                    "citations": ";".join(citations),
                    "factual_correctness": "",
                    "cited_passage_support": "",
                    "reviewer": "",
                    "notes": "",
                }
            )

        relevance = retrieval_metrics(rankings, relevant)
        config_results.append(
            {
                "chunk_size_words": chunk_size,
                "overlap_words": overlap,
                "index": {"chunk_count": len(chunks), "vector_count": len(store), "dimension": store.dimension},
                "timings": {
                    "document_load_seconds": document_load_seconds,
                    "chunking_seconds": chunking_seconds,
                    "document_embedding_seconds": document_embedding_seconds,
                    "query_embedding_seconds": latency_summary(query_embedding_samples),
                    "faiss_retrieval_seconds": latency_summary(retrieval_samples),
                    "ollama_generation_seconds": latency_summary(generation_samples),
                    "total_query_pipeline_seconds": latency_summary(total_query_samples),
                },
                "retrieval_metrics": relevance,
                "answerability": answerability_metrics(answerable, predicted_answerable),
                "queries": rows,
            }
        )

    return {
        "benchmark": {
            "title": dataset.get("title"),
            "version": dataset.get("version"),
            "label_type": dataset.get("label_type"),
            "question_count": len(questions),
            "document_count": len(paths),
            "passage_annotation_count": len(passage_annotations),
        },
        "method": {
            "embedding_model": embedding_model,
            "embedding_model_initialization_seconds": model_initialization_seconds,
            "vector_dimension": embedder.dimension,
            "index_type": "FAISS IndexFlatIP over normalized float32 vectors",
            "minimum_score": minimum_score,
            "score_interpretation": "cosine similarity, not probability of relevance",
            "generation_enabled": with_generation,
            "generation_model": ollama_model if with_generation else None,
            "chunk_labels": "stable source passage anchors projected onto each chunk by word-span overlap",
            "human_review_performed": False,
            "answerability_prediction_basis": "generated status" if with_generation else "retrieval produced at least one chunk",
        },
        "runtime": {
            "python": platform.python_version(),
            "packages": {
                package: _package_version(distribution)
                for package, distribution in {
                    "PyMuPDF": "PyMuPDF",
                    "sentence-transformers": "sentence-transformers",
                    "FAISS": "faiss-cpu",
                    "NumPy": "numpy",
                }.items()
            },
        },
        "configurations": config_results,
        "review_rows": review_rows,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )


def _markdown_report(results: dict[str, object]) -> str:
    benchmark = results["benchmark"]
    method = results["method"]
    runtime = results["runtime"]
    lines = [
        "# LocalRAG Evaluation Report",
        "",
        f"- Dataset: {benchmark['title']} (v{benchmark['version']}); {benchmark['question_count']} questions, {benchmark['document_count']} synthetic text documents.",
        f"- Labels: `{benchmark['label_type']}`. These are fixture labels, not reviews of external documents.",
        f"- Embedding model: `{method['embedding_model']}` ({method['vector_dimension']} dimensions).",
        f"- Runtime: Python {runtime['python']}; package versions `{runtime['packages']}`.",
        f"- Index: {method['index_type']}.",
        f"- Score threshold: {method['minimum_score']}; score is cosine similarity, not a probability.",
        f"- Model initialization measured separately: {method['embedding_model_initialization_seconds']:.4f} seconds (may be warm or cached; this run does not claim a cold start).",
        f"- Ollama answer generation enabled: {method['generation_enabled']}; human review performed: no.",
        "",
        "## Results",
        "",
        "| Chunk size / overlap | Chunks | Dimension | Recall@1 | Recall@3 | Recall@5 | Precision@1 | Precision@3 | Precision@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in results["configurations"]:
        retrieval = config["retrieval_metrics"]
        timings = config["timings"]

        def display(value: object) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        lines.append(
            "| {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                config["chunk_size_words"], config["overlap_words"],
                config["index"]["chunk_count"],
                config["index"]["dimension"],
                display(retrieval["recall_at_k"]["1"]),
                display(retrieval["recall_at_k"]["3"]),
                display(retrieval["recall_at_k"]["5"]),
                display(retrieval["precision_at_k"]["1"]),
                display(retrieval["precision_at_k"]["3"]),
                display(retrieval["precision_at_k"]["5"]),
                display(retrieval["mrr"]),
            )
        )
    lines.append("")
    for config in results["configurations"]:
        timings = config["timings"]

        def display(value: object) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        def display_seconds(value: object) -> str:
            return "n/a" if value is None else f"{value:.4f}s"

        for key, label in (
            ("query_embedding_seconds", "query embedding"),
            ("faiss_retrieval_seconds", "FAISS retrieval"),
            ("ollama_generation_seconds", "Ollama generation"),
            ("total_query_pipeline_seconds", "total measured query"),
        ):
            summary = timings[key]
            lines.append(
                f"- {config['chunk_size_words']}/{config['overlap_words']} {label}: "
                f"n={summary['count']}, mean={display_seconds(summary['mean'])}, "
                f"p50={display_seconds(summary['p50'])}, p95={display_seconds(summary['p95'])}."
            )
        lines.append(
            f"- {config['chunk_size_words']}/{config['overlap_words']} ingestion: "
            f"document load={timings['document_load_seconds']:.4f}s, "
            f"chunking={timings['chunking_seconds']:.4f}s, "
            f"batch document embedding={timings['document_embedding_seconds']:.4f}s."
        )
    lines.extend(
        [
            "",
            "Retrieval metrics are averaged over queries with non-empty gold passage annotations only. Unsupported and missing-label queries are counted separately. Precision@k divides by k, including when fewer than k passage results exist. MRR uses the first relevant passage. Latency p95 is emitted only for at least 20 samples.",
            "",
            f"Answerability predictions are based on `{method['answerability_prediction_basis']}`.",
            "",
            "## Answerability and citations",
            "",
        ]
    )
    for config in results["configurations"]:
        answerability = config["answerability"]
        lines.append(
            f"- {config['chunk_size_words']}/{config['overlap_words']}: {answerability['unsupported_answer_count']} unsupported non-abstentions, {answerability['supported_abstention_count']} supported abstentions, {answerability['correct_abstention_count']} correct abstentions. Citation validity indicates label membership only; it is not a grounding judgment."
        )
    lines.extend(
        [
            "",
            "## Method and limits",
            "",
            "The benchmark corpus is synthetic and included in this repository. Stable passage anchors are projected onto chunks by word-span overlap, so gold labels survive both chunk boundaries. Both chunk configurations use the same documents and embedding model. Query embedding, FAISS search, optional generation, and total measured query time are reported separately; model initialization is separate from query timings. Process startup and Python imports are excluded.",
            "",
            "Similarity thresholds trade false positives (irrelevant chunks pass) against false negatives (relevant chunks are filtered). The default threshold is disabled because this small synthetic set does not justify a general cutoff. Optional generation requires a locally running Ollama model. No human answer review is implied by these automated metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LocalRAG retrieval on the synthetic fixture benchmark.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--passages", type=Path, default=DEFAULT_PASSAGES)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum-score", type=float, default=None)
    parser.add_argument("--with-generation", action="store_true")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--api-url", default="http://localhost:11434/api/chat")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "latest.json")
    parser.add_argument("--csv-output", type=Path, default=ROOT / "results" / "latest.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "report.md")
    parser.add_argument("--review-template", type=Path, default=ROOT / "results" / "human_review.csv")
    args = parser.parse_args(argv)
    try:
        if args.top_k <= 0:
            raise ValueError("top-k must be positive")
        if args.minimum_score is not None and not -1 <= args.minimum_score <= 1:
            raise ValueError("minimum-score must be between -1 and 1")
        results = evaluate(
            dataset_path=args.dataset,
            passages_path=args.passages,
            documents_dir=args.documents,
            embedding_model=args.embedding_model,
            top_k=args.top_k,
            minimum_score=args.minimum_score,
            with_generation=args.with_generation,
            ollama_model=args.ollama_model,
            api_url=args.api_url,
            timeout=args.timeout,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        flat_rows = [
            {"chunk_size": config["chunk_size_words"], "overlap": config["overlap_words"], **row}
            for config in results["configurations"] for row in config["queries"]
        ]
        _write_csv(args.csv_output, flat_rows, list(flat_rows[0]) if flat_rows else [])
        _write_csv(
            args.review_template,
            results["review_rows"],
            ["chunk_size", "overlap", "question_id", "question", "answer", "citations", "factual_correctness", "cited_passage_support", "reviewer", "notes"],
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(_markdown_report(results), encoding="utf-8")
    except Exception as error:
        print(f"evaluation: {error}", file=sys.stderr)
        return 1
    print(f"Wrote JSON: {args.output}")
    print(f"Wrote CSV: {args.csv_output}")
    print(f"Wrote report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
