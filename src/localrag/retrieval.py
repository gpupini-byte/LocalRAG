# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Exact cosine-similarity retrieval backed by FAISS IndexFlatIP."""

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from numpy.typing import ArrayLike, NDArray

from localrag.chunker import ChunkRecord
from localrag.embeddings import EmbeddingBatch
from localrag.embeddings import DEFAULT_MODEL_NAME

if TYPE_CHECKING:
    from localrag.embeddings import SentenceTransformerEmbedder


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A retrieved chunk and its cosine similarity score."""

    chunk: ChunkRecord
    score: float


class FaissVectorStore:
    """Exact cosine search with optional generation-based persistence.

    FAISS index files are binary inputs and must only be loaded from trusted
    sources. Metadata is JSON and is validated against the loaded FAISS index.
    """

    _SCHEMA_VERSION = 1

    def __init__(
        self, dimension: int = 384, model_name: str = DEFAULT_MODEL_NAME
    ) -> None:
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        self.dimension = dimension
        self.model_name = model_name
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: list[ChunkRecord] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> tuple[ChunkRecord, ...]:
        """Read-only chunk metadata aligned with FAISS vector positions."""
        return tuple(self._chunks)

    def save(self, directory: str | Path) -> None:
        """Persist an immutable index generation and atomically select it.

        The directory contains ``manifest.json`` and a ``generations`` folder.
        An interrupted write can leave an unused generation, but cannot make
        the manifest point to a partially written generation.
        """
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        generations = root / "generations"
        generations.mkdir(exist_ok=True)

        generation_id = uuid.uuid4().hex
        temporary_generation = generations / f".tmp-{generation_id}"
        final_generation = generations / generation_id
        temporary_generation.mkdir()
        index_path = temporary_generation / "index.faiss"
        metadata_path = temporary_generation / "metadata.json"

        try:
            faiss.write_index(self._index, str(index_path))
            metadata = {
                "schema_version": self._SCHEMA_VERSION,
                "embedding_model": self.model_name,
                "dimension": self.dimension,
                "count": len(self._chunks),
                "chunks": [
                    {
                        "vector_position": row,
                        "id": chunk.id,
                        "text": chunk.text,
                        "source": chunk.source,
                        "page": chunk.page,
                        "position": chunk.position,
                    }
                    for row, chunk in enumerate(self._chunks)
                ],
            }
            self._write_json(metadata_path, metadata)
            # Read back the staged generation before exposing it through the
            # manifest. A failed/partial write leaves the previous generation
            # selected and searchable.
            staged_index = faiss.read_index(str(index_path))
            staged_metadata = self._read_json(metadata_path, "staged metadata")
            if (staged_index.d != self.dimension or staged_index.ntotal != len(self._chunks)
                    or staged_index.metric_type != faiss.METRIC_INNER_PRODUCT):
                raise ValueError("Staged FAISS generation failed dimension/count/metric validation")
            if (staged_metadata.get("schema_version") != self._SCHEMA_VERSION
                    or staged_metadata.get("dimension") != self.dimension
                    or staged_metadata.get("count") != len(self._chunks)
                    or staged_metadata.get("embedding_model") != self.model_name
                    or not isinstance(staged_metadata.get("chunks"), list)
                    or len(staged_metadata["chunks"]) != len(self._chunks)):
                raise ValueError("Staged FAISS metadata failed schema/count/model validation")
            staged_chunks = [self._chunk_from_json(item, row)
                             for row, item in enumerate(staged_metadata["chunks"])]
            if len({chunk.id for chunk in staged_chunks}) != len(staged_chunks):
                raise ValueError("Staged FAISS metadata contains duplicate chunk IDs")
            os.replace(temporary_generation, final_generation)

            manifest_tmp = root / f".manifest-{generation_id}.tmp"
            self._write_json(
                manifest_tmp,
                {"schema_version": self._SCHEMA_VERSION, "generation": generation_id},
            )
            os.replace(manifest_tmp, root / "manifest.json")
        except Exception:
            # Keep completed generations for recovery/inspection; remove only
            # a still-temporary directory that was never made active.
            if temporary_generation.exists():
                for child in temporary_generation.iterdir():
                    child.unlink()
                temporary_generation.rmdir()
            raise

    @staticmethod
    def _write_json(path: Path, content: dict[str, object]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(content, file, ensure_ascii=False, separators=(",", ":"))
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

    @classmethod
    def load(
        cls,
        directory: str | Path,
        expected_model_name: str | None = None,
    ) -> "FaissVectorStore":
        """Load a saved generation after validating JSON and FAISS agreement.

        Only load FAISS index files from trusted sources. FAISS binary index
        files may be unsafe when supplied by an untrusted party.
        """
        root = Path(directory)
        manifest_path = root / "manifest.json"
        try:
            manifest = cls._read_json(manifest_path, "manifest")
            cls._validate_keys(manifest, {"schema_version", "generation"}, "manifest")
            if (
                not isinstance(manifest["schema_version"], int)
                or isinstance(manifest["schema_version"], bool)
                or manifest["schema_version"] != cls._SCHEMA_VERSION
            ):
                raise ValueError(f"Unsupported manifest schema version: {manifest['schema_version']!r}")
            generation_id = manifest["generation"]
            if not isinstance(generation_id, str) or not re.fullmatch(r"[0-9a-f]{32}", generation_id):
                raise ValueError("Invalid generation identifier in manifest")
        except OSError as error:
            raise ValueError(f"Could not read index manifest at '{manifest_path}': {error}") from error

        generation = root / "generations" / generation_id
        metadata_path = generation / "metadata.json"
        index_path = generation / "index.faiss"
        try:
            metadata = cls._read_json(metadata_path, "metadata")
        except OSError as error:
            raise ValueError(f"Could not read index metadata at '{metadata_path}': {error}") from error

        required = {"schema_version", "embedding_model", "dimension", "count", "chunks"}
        cls._validate_keys(metadata, required, "metadata")
        if (
            not isinstance(metadata["schema_version"], int)
            or isinstance(metadata["schema_version"], bool)
            or metadata["schema_version"] != cls._SCHEMA_VERSION
        ):
            raise ValueError(f"Unsupported metadata schema version: {metadata['schema_version']!r}")
        model_name = metadata["embedding_model"]
        dimension = metadata["dimension"]
        count = metadata["count"]
        raw_chunks = metadata["chunks"]
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Metadata embedding_model must be a non-empty string")
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
            raise ValueError("Metadata dimension must be a positive integer")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("Metadata count must be a non-negative integer")
        if not isinstance(raw_chunks, list) or len(raw_chunks) != count:
            raise ValueError("Metadata count does not match the number of chunk records")
        if expected_model_name is not None and model_name != expected_model_name:
            raise ValueError(
                f"Embedding model mismatch: index uses '{model_name}', "
                f"expected '{expected_model_name}'"
            )

        chunks = [cls._chunk_from_json(item, position) for position, item in enumerate(raw_chunks)]
        ids = [chunk.id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Metadata contains duplicate chunk IDs")

        try:
            index = faiss.read_index(str(index_path))
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError(f"Could not read FAISS index at '{index_path}': {error}") from error
        if index.d != dimension:
            raise ValueError(
                f"FAISS index dimension {index.d} does not match metadata dimension {dimension}"
            )
        if index.ntotal != count:
            raise ValueError(
                f"FAISS index count {index.ntotal} does not match metadata count {count}"
            )
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("FAISS index metric does not match cosine-similarity IndexFlatIP")

        store = cls(dimension=dimension, model_name=model_name)
        store._index = index
        store._chunks = chunks
        return store

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, object]:
        try:
            with path.open("r", encoding="utf-8") as file:
                value = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Corrupted {label} JSON at '{path}': {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{label.capitalize()} JSON must contain an object")
        return value

    @staticmethod
    def _validate_keys(value: dict[str, object], expected: set[str], label: str) -> None:
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(f"Invalid {label} schema (missing keys: {missing}; extra keys: {extra})")

    @classmethod
    def _chunk_from_json(cls, value: object, row: int) -> ChunkRecord:
        if not isinstance(value, dict):
            raise ValueError(f"Metadata chunk at row {row} must be an object")
        cls._validate_keys(
            value,
            {"vector_position", "id", "text", "source", "page", "position"},
            f"chunk row {row}",
        )
        vector_position = value["vector_position"]
        if not isinstance(vector_position, int) or isinstance(vector_position, bool) or vector_position != row:
            raise ValueError(f"Metadata vector position at row {row} is inconsistent")
        chunk_id, text, source = value["id"], value["text"], value["source"]
        page, position = value["page"], value["position"]
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(f"Metadata chunk at row {row} has an invalid id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Metadata chunk at row {row} has empty text")
        if not isinstance(source, str):
            raise ValueError(f"Metadata chunk at row {row} has an invalid source")
        if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page <= 0):
            raise ValueError(f"Metadata chunk at row {row} has an invalid page number")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError(f"Metadata chunk at row {row} has an invalid position")
        return ChunkRecord(text=text, source=source, page=page, id=chunk_id, position=position)

    def add(self, batch: EmbeddingBatch) -> None:
        """Add an embedding batch, normalizing rows for cosine search."""
        vectors = np.asarray(batch.embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape != (len(batch.chunks), self.dimension):
            raise ValueError(
                f"Embedding batch shape {vectors.shape} does not match "
                f"({len(batch.chunks)}, {self.dimension})"
            )
        if not batch.chunks:
            return
        if not np.isfinite(vectors).all():
            raise ValueError("Embedding batch contains non-finite values")

        vectors = np.ascontiguousarray(vectors.copy())
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            raise ValueError("Embedding batch contains a zero-length vector")
        vectors /= norms[:, None]
        self._index.add(vectors)
        self._chunks.extend(batch.chunks)

    def search(self, query_vector: ArrayLike, top_k: int = 5) -> list[SearchResult]:
        """Return the most similar chunks for one query embedding."""
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if top_k == 0 or not self._chunks:
            return []

        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query[0]
        if query.shape != (self.dimension,):
            raise ValueError(f"query vector must have shape ({self.dimension},)")
        if not np.isfinite(query).all():
            raise ValueError("query vector contains non-finite values")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("query vector must have non-zero length")

        query = np.ascontiguousarray((query / norm).reshape(1, -1), dtype=np.float32)
        scores, indices = self._index.search(query, min(top_k, len(self._chunks)))
        return [
            SearchResult(chunk=self._chunks[int(index)], score=float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def search_text(
        self,
        query: str,
        embedder: "SentenceTransformerEmbedder",
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Embed a text query and return its nearest chunks."""
        if not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer")
        if not query or not query.strip():
            raise ValueError("query must not be empty or whitespace-only")
        if embedder.dimension != self.dimension:
            raise ValueError(
                f"Embedder dimension {embedder.dimension} does not match index dimension {self.dimension}"
            )
        if top_k == 0 or not self._chunks:
            return []

        query_chunk = ChunkRecord(
            text=query,
            source="<query>",
            page=None,
            id="<query>",
            position=0,
        )
        query_batch = embedder.embed([query_chunk])
        return self.search(query_batch.embeddings[0], top_k=top_k)
