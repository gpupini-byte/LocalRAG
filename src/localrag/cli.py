# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""CLI commands for ingesting local documents and querying LocalRAG."""

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from localrag.chunker import chunk_documents
from localrag.generator import OllamaChatClient
from localrag.loader import load_document
from localrag.pipeline import LocalRAGPipeline
from localrag.retrieval import FaissVectorStore


class UninitializedIndexError(RuntimeError):
    """The user needs to ingest documents before running a query."""


def _ingest_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m localrag.cli ingest",
        description="Load PDF/TXT files, embed chunks, and persist a FAISS index.",
    )
    parser.add_argument("--documents", type=Path, default=Path("data/documents"))
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    return parser


def _query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask a question over ingested local documents.")
    parser.add_argument("question", help="Question to answer from the indexed documents")
    parser.add_argument("--documents", type=Path, default=Path("data/documents"),
                        help="Kept for CLI compatibility; documents are ingested separately")
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--ollama-model", default="llama3.2:3b")
    parser.add_argument("--api-url", default="http://localhost:11434/api/chat")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--minimum-score",
        type=float,
        default=None,
        help="optional cosine similarity cutoff (disabled by default; not a relevance probability)",
    )
    return parser


def _citation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search indexed documents by topic and format verified citations.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-chunks", type=int, default=None)
    parser.add_argument("--minimum-score", type=float, default=None)
    parser.add_argument("--crossref", action="store_true", help="Allow optional Crossref metadata lookup")
    parser.add_argument("--crossref-timeout", type=float, default=5.0)
    parser.add_argument("--metadata-store", type=Path, default=Path("data/bibliography/records.json"))
    return parser


def _citation_main(argv: Sequence[str]) -> int:
    args = _citation_parser().parse_args(list(argv))
    try:
        store = _load_index(args.index, args.embedding_model)
        from localrag.embeddings import SentenceTransformerEmbedder
        from localrag.citations import CrossrefClient, search_citations
        from localrag.metadata_review import BibliographicMetadataStore
        embedder = SentenceTransformerEmbedder(args.embedding_model)
        crossref = CrossrefClient(timeout=args.crossref_timeout) if args.crossref else None
        results = search_citations(args.query, embedder, store, args.top_k,
                                   args.candidate_chunks, args.minimum_score, crossref=crossref)
        metadata_store = BibliographicMetadataStore(args.metadata_store)
        from dataclasses import replace
        reviewed_results = []
        for item in results:
            record = metadata_store.observe(item)
            reviewed_results.append(replace(item, metadata=record.effective_metadata(), status=record.status))
        results = reviewed_results
    except Exception as error:
        print(f"localrag: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"score_semantics": "cosine similarity, not probability or citation correctness",
                      "results": [{"rank": item.rank, "document_id": item.document_id,
                                   "title": item.title, "cosine_similarity": item.score,
                                   "passage": item.passage, "source_file": item.source,
                                   "page": item.page, "doi": item.metadata.doi,
                                   "mla_works_cited": item.works_cited,
                                   "metadata_status": item.status,
                                   "missing_fields": list(item.metadata.missing_fields),
                                   "field_provenance": item.metadata.provenance,
                                   "field_confidence": item.metadata.confidence,
                                   "verified_fields": list(item.metadata.verified_fields),
                                   "pdf_metadata": item.metadata.original_pdf_metadata,
                                   "pdf_candidates": item.metadata.pdf_candidates,
                                   "candidate_matches": [{"title": c.metadata.title,
                                                          "authors": list(c.metadata.authors),
                                                          "year": c.metadata.year, "doi": c.metadata.doi,
                                                          "container": c.metadata.container,
                                                          "score": c.score} for c in item.candidates]}
                                  for item in results]}, ensure_ascii=True, indent=2))
    return 0


def _citation_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect and review saved bibliographic metadata.")
    parser.add_argument("action", choices=("list", "show", "approve", "mla"))
    parser.add_argument("identifier", nargs="?", help="Document ID or DOI-backed record key")
    parser.add_argument("--store", type=Path, default=Path("data/bibliography/records.json"))
    parser.add_argument("--status", choices=("verified", "ambiguous", "incomplete", "manually_reviewed"))
    parser.add_argument("--field", choices=("title", "authors", "year", "container", "journal", "volume", "issue", "pages", "doi", "kind"))
    parser.add_argument("--value-json", help="Approved value as JSON (authors must be a JSON list)")
    parser.add_argument("--source-url", help="Source URL used to verify this correction")
    parser.add_argument("--reviewer", help="Reviewer name or identifier")
    return parser


def _citation_review_main(argv: Sequence[str]) -> int:
    args = _citation_review_parser().parse_args(list(argv))
    try:
        from dataclasses import asdict
        from localrag.metadata_review import BibliographicMetadataStore
        store = BibliographicMetadataStore(args.store)
        if args.action == "list":
            records = store.list_records(args.status)
            payload = [{"key": row.key, "document_ids": row.document_ids, "sources": row.sources,
                        "status": row.status, "fields": row.fields, "candidates": row.candidates,
                        "original_pdf_metadata": row.original_pdf_metadata}
                       for row in records]
        elif not args.identifier:
            raise ValueError(f"{args.action} requires a document ID or DOI record key")
        elif args.action == "show":
            payload = asdict(store.get(args.identifier))
        elif args.action == "approve":
            if not all((args.field, args.value_json, args.source_url, args.reviewer)):
                raise ValueError("approve requires --field, --value-json, --source-url, and --reviewer")
            try:
                value = json.loads(args.value_json)
            except json.JSONDecodeError as error:
                raise ValueError(f"--value-json is invalid JSON: {error}") from error
            record = store.approve(args.identifier, args.field, value, args.source_url, args.reviewer)
            payload = {"key": record.key, "status": record.status, "fields": record.fields,
                       "overrides": record.overrides, "works_cited": record.works_cited}
        else:
            record = store.get(args.identifier)
            payload = {"key": record.key, "status": record.status, "works_cited": record.works_cited}
            if record.works_cited is None:
                payload["message"] = "Citation is incomplete or unverified; approve required fields before generating MLA."
        print(json.dumps(payload, ensure_ascii=True, indent=2))
        return 0
    except Exception as error:
        print(f"localrag: {error}", file=sys.stderr)
        return 1


def _ingest_documents(
    documents_dir: Path,
    index_dir: Path,
    embedding_model: str,
    embedder: object | None = None,
) -> tuple[FaissVectorStore, int, int]:
    """Create a new active index generation from current supported documents."""
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    if embedder is None:
        from localrag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(embedding_model)
    elif getattr(embedder, "model_name", embedding_model) != embedding_model:
        raise ValueError("Embedder model does not match requested embedding model")

    paths = [
        path
        for path in sorted(documents_dir.rglob("*"))
        if path.is_file() and path.suffix.casefold() in {".pdf", ".txt"}
    ]
    records = []
    for path in paths:
        records.extend(load_document(path))
    chunks = chunk_documents(records)
    if not chunks:
        raise ValueError(
            f"No non-empty PDF/TXT text found in '{documents_dir}'; no index was written."
        )

    batch = embedder.embed(chunks)
    dimension = embedder.dimension
    store = FaissVectorStore(dimension=dimension, model_name=embedding_model)
    store.add(batch)
    store.save(index_dir)
    return store, len(paths), len(chunks)


def _load_index(index_dir: Path, embedding_model: str) -> FaissVectorStore:
    manifest = index_dir / "manifest.json"
    if not manifest.is_file():
        raise UninitializedIndexError(
            f"No LocalRAG index found at '{index_dir}'. Run "
            "`python -m localrag.cli ingest` before asking a question."
        )
    store = FaissVectorStore.load(index_dir, expected_model_name=embedding_model)
    if len(store) == 0:
        raise UninitializedIndexError(
            f"The index at '{index_dir}' contains no vectors. Run "
            "`python -m localrag.cli ingest` after adding non-empty PDF/TXT files."
        )
    return store


def _ingest_main(argv: Sequence[str]) -> int:
    args = _ingest_parser().parse_args(list(argv))
    try:
        store, source_count, chunk_count = _ingest_documents(
            args.documents, args.index, args.embedding_model
        )
    except Exception as error:
        print(f"localrag: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "ingested",
                "sources": source_count,
                "chunks": chunk_count,
                "vectors": len(store),
                "dimension": store.dimension,
                "embedding_model": store.model_name,
                "index": str(args.index),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _query_main(argv: Sequence[str]) -> int:
    args = _query_parser().parse_args(list(argv))
    try:
        store = _load_index(args.index, args.embedding_model)
        from localrag.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder(args.embedding_model)
        pipeline = LocalRAGPipeline(
            embedder=embedder,
            vector_store=store,
            generator=OllamaChatClient(
                model=args.ollama_model, api_url=args.api_url, timeout=args.timeout
            ),
            top_k=args.top_k,
            minimum_score=args.minimum_score,
        )
        response = pipeline.answer(args.question)
    except Exception as error:
        print(f"localrag: {error}", file=sys.stderr)
        return 1

    payload = {
        "status": response.status,
        "answer": response.answer,
        "citations": list(response.citations),
        "evidence": [
            {
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "page": item.page,
                "score": item.score,
            }
            for item in response.evidence
        ],
        "timings_seconds": response.timings,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "ingest":
        return _ingest_main(args[1:])
    if args and args[0] == "search-citations":
        return _citation_main(args[1:])
    if args and args[0] == "citation-review":
        return _citation_review_main(args[1:])
    return _query_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
