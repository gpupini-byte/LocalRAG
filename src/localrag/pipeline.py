# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""End-to-end retrieval, grounded prompting, and answer generation."""

import re
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from localrag.chunker import ChunkRecord
from localrag.embeddings import EmbeddingBatch
from localrag.generator import (
    ChatMessage,
    InvalidOllamaResponseError,
    OllamaChatClient,
)
from localrag.prompting import EvidenceItem, build_grounded_messages, make_evidence
from localrag.retrieval import FaissVectorStore


_CITATION_PATTERN = re.compile(r"\[(S\d+)\]")
_INSUFFICIENT_ANSWER = "I don't have enough evidence in the indexed documents to answer that question."
_ABSTENTION_MARKERS = (
    "insufficient evidence", "evidence is insufficient", "not enough evidence",
    "cannot answer from the provided", "does not support an answer",
    "not supported by the provided evidence", "not supported by the provided passages",
    "indexed evidence is insufficient", "no passage supports",
    "not directly supported by the provided", "not supported by the provided",
    "no mention of",
)


class QueryEmbedder(Protocol):
    dimension: int

    def embed(self, chunks: list[ChunkRecord]) -> EmbeddingBatch: ...


class AnswerGenerator(Protocol):
    def generate(self, messages: list[ChatMessage]) -> str: ...


@dataclass(frozen=True, slots=True)
class GroundedResponse:
    answer: str
    citations: tuple[str, ...]
    evidence: tuple[EvidenceItem, ...]
    timings: dict[str, float]
    status: str
    scope_note: str | None = None


class LocalRAGPipeline:
    """Coordinate query embedding, FAISS retrieval, and Ollama generation."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: FaissVectorStore,
        generator: AnswerGenerator | None = None,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> None:
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if minimum_score is not None and not -1.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between -1 and 1")
        if embedder.dimension != vector_store.dimension:
            raise ValueError(
                f"Embedder dimension {embedder.dimension} does not match "
                f"index dimension {vector_store.dimension}"
            )
        embedder_model = getattr(embedder, "model_name", None)
        if embedder_model is not None and embedder_model != vector_store.model_name:
            raise ValueError(
                f"Embedder model '{embedder_model}' does not match index model "
                f"'{vector_store.model_name}'"
            )
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator or OllamaChatClient()
        self.top_k = top_k
        self.minimum_score = minimum_score

    def answer(self, query: str) -> GroundedResponse:
        """Answer a query using only the retrieved document chunks as evidence."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        total_start = perf_counter()
        timings = {"embedding": 0.0, "retrieval": 0.0, "generation": 0.0}

        if len(self.vector_store) == 0:
            timings["total"] = perf_counter() - total_start
            return GroundedResponse(
                answer=_INSUFFICIENT_ANSWER,
                citations=(),
                evidence=(),
                timings=timings,
                status="insufficient_evidence",
            )

        query_chunk = ChunkRecord(
            text=query.strip(), source="<query>", page=None, id="<query>", position=0
        )
        start = perf_counter()
        query_batch = self.embedder.embed([query_chunk])
        timings["embedding"] = perf_counter() - start

        broad_question = _is_broad_library_question(query)
        explicit_anchors = _explicit_query_anchors(query)
        numeric_anchors = _numeric_query_anchors(query)
        start = perf_counter()
        expand_pool = broad_question or bool(explicit_anchors or numeric_anchors)
        pool_size = min(len(self.vector_store), max(self.top_k * 5, 20)) if expand_pool else self.top_k
        pool = self.vector_store.search(query_batch.embeddings[0], top_k=pool_size)
        if self.minimum_score is not None:
            pool = [result for result in pool if result.score >= self.minimum_score]
        required_anchors = (*explicit_anchors, *numeric_anchors)
        if required_anchors:
            anchored = [result for result in pool
                        if all(anchor in result.chunk.text.casefold() for anchor in required_anchors)]
            if anchored:
                pool = anchored
            elif numeric_anchors:
                # Exact numbers in a question are identifiers or values. If no
                # retrieved text contains them, do not let semantic neighbors
                # stand in for evidence about that exact number.
                pool = []
        if _is_cpu_focused_question(query):
            cpu_focused = [result for result in pool
                           if _is_cpu_focused_passage(result.chunk.text)]
            # Keep the original pool if extraction happens to omit CPU
            # terminology; topical filtering must never erase all evidence.
            if cpu_focused:
                pool = cpu_focused
        if broad_question:
            diverse: list = []
            seen_sources: set[str] = set()
            for result in pool:
                identity = str(Path(result.chunk.source).resolve(strict=False))
                if identity not in seen_sources:
                    diverse.append(result)
                    seen_sources.add(identity)
            results = (diverse + [item for item in pool if item not in diverse])[:self.top_k]
        else:
            results = pool[:self.top_k]
        if self.minimum_score is not None:
            results = [result for result in results if result.score >= self.minimum_score]
        timings["retrieval"] = perf_counter() - start
        if not results:
            timings["total"] = perf_counter() - total_start
            return GroundedResponse(
                answer=_INSUFFICIENT_ANSWER,
                citations=(),
                evidence=(),
                timings=timings,
                status="insufficient_evidence",
                scope_note=_broad_scope_note() if broad_question else None,
            )

        evidence = make_evidence(results)
        messages = build_grounded_messages(query.strip(), evidence)
        start = perf_counter()
        answer = self.generator.generate(messages)
        timings["generation"] = perf_counter() - start
        if not isinstance(answer, str) or not answer.strip():
            raise InvalidOllamaResponseError("Ollama returned an empty answer")

        allowed_ids = {item.source_id for item in evidence}
        cited_ids = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
        if any(source_id not in allowed_ids for source_id in cited_ids):
            timings["total"] = perf_counter() - total_start
            return GroundedResponse(
                answer=(
                    "I couldn't return a grounded answer because the model cited a source "
                    "that was not included in the retrieved evidence."
                ),
                citations=(),
                evidence=tuple(evidence),
                timings=timings,
                status="invalid_citations",
            )
        if not cited_ids:
            # One bounded formatting retry addresses citation-label omissions.
            retry_messages = [*messages, {
                "role": "user",
                "content": ("Your previous response omitted valid source labels. Revise it once. "
                            "Answer any part that the evidence does support, even if the evidence "
                            "does not support the whole explanation; state the limitation clearly. "
                            "Abstain only when no meaningful part is supported. Cite every factual "
                            "claim using exact source IDs in the supplied evidence, such as [S1]. "
                            "Do not add sources or unsupported claims."),
            }]
            retry_start = perf_counter()
            answer = self.generator.generate(retry_messages)
            timings["generation"] += perf_counter() - retry_start
            if not isinstance(answer, str) or not answer.strip():
                raise InvalidOllamaResponseError("Ollama returned an empty answer on citation retry")
            cited_ids = tuple(dict.fromkeys(_CITATION_PATTERN.findall(answer)))
            if any(source_id not in allowed_ids for source_id in cited_ids):
                timings["total"] = perf_counter() - total_start
                return GroundedResponse(
                    "I couldn't return a grounded answer because the model cited a source that was not included in the retrieved evidence.",
                    (), tuple(evidence), timings, "invalid_citations",
                    _broad_scope_note() if broad_question else None)
            if not cited_ids and _declares_insufficient_evidence(answer):
                timings["total"] = perf_counter() - total_start
                return GroundedResponse(answer.strip(), (), tuple(evidence), timings,
                                        "insufficient_evidence",
                                        _broad_scope_note() if broad_question else None)
            if cited_ids:
                timings["total"] = perf_counter() - total_start
                return GroundedResponse(answer.strip(), cited_ids, tuple(evidence), timings, "ok",
                                        _broad_scope_note() if broad_question else None)
            timings["total"] = perf_counter() - total_start
            return GroundedResponse(
                answer="I couldn't return a grounded answer because the model did not cite the retrieved evidence.",
                citations=(),
                evidence=tuple(evidence),
                timings=timings,
                status="missing_citations",
                scope_note=_broad_scope_note() if broad_question else None,
            )

        timings["total"] = perf_counter() - total_start
        return GroundedResponse(
            answer=answer.strip(),
            citations=cited_ids,
            evidence=tuple(evidence),
            timings=timings,
            status="ok",
            scope_note=_broad_scope_note() if broad_question else None,
        )


def _is_broad_library_question(query: str) -> bool:
    normalized = re.sub(r"\s+", " ", query.casefold()).strip()
    return any(phrase in normalized for phrase in (
        "those papers", "these papers", "all papers", "all documents", "my library",
        "across the papers", "across my documents", "summarize what", "summarise what",
        "resume what", "summarize these", "summarise these", "overview of the papers",
    ))


def _is_cpu_focused_question(query: str) -> bool:
    normalized = query.casefold()
    return bool(re.search(r"\b(cpu|processor)\b", normalized)) and not bool(
        re.search(r"\bgpu\b", normalized)
    )


def _explicit_query_anchors(query: str) -> tuple[str, ...]:
    """Find literal acronyms and CamelCase product names in a query."""
    anchors = re.findall(r"\b[A-Z]{2,}\b|\b[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b", query)
    return tuple(dict.fromkeys(item.casefold() for item in anchors))


def _numeric_query_anchors(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"\b\d+\b", query)))


def _is_cpu_focused_passage(text: str) -> bool:
    normalized = text.casefold()
    cpu_mentions = len(re.findall(r"\b(cpu|cpus|processor|processors)\b", normalized))
    gpu_mentions = len(re.findall(r"\b(gpu|gpus)\b", normalized))
    return cpu_mentions > 0 and cpu_mentions >= gpu_mentions


def _declares_insufficient_evidence(answer: str) -> bool:
    normalized = re.sub(r"\s+", " ", answer.casefold()).strip()
    return any(marker in normalized for marker in _ABSTENTION_MARKERS)


def _broad_scope_note() -> str:
    return ("Broad library question: this answer uses only the top retrieved passages and is not a "
            "complete summary of every indexed paper. Narrow the topic or review individual sources for a fuller survey.")
