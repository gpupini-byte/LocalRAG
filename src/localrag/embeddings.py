# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Sentence-transformer embeddings for LocalRAG chunks."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

from localrag.chunker import ChunkRecord


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass(frozen=True, slots=True)
class EmbeddingBatch:
    """Vectors aligned row-for-row with the original chunks tuple."""

    embeddings: NDArray[np.float32]
    chunks: tuple[ChunkRecord, ...]


class SentenceTransformerEmbedder:
    """Encode chunks into normalized vectors with one model per instance."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, *, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, local_files_only=local_files_only)
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            # Existing fake/older compatible models may only expose the former API.
            get_dimension = getattr(self._model, "get_sentence_embedding_dimension", None)
        dimension = get_dimension() if get_dimension is not None else None
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(
                f"Model '{model_name}' reported an invalid embedding dimension: {dimension!r}"
            )
        self.dimension = dimension

    def embed(self, chunks: Sequence[ChunkRecord]) -> EmbeddingBatch:
        """Embed a batch, preserving chunk order and metadata alongside rows."""
        ordered_chunks = tuple(chunks)
        if not ordered_chunks:
            return EmbeddingBatch(
                embeddings=np.empty((0, self.dimension), dtype=np.float32),
                chunks=ordered_chunks,
            )

        for index, chunk in enumerate(ordered_chunks):
            if not chunk.text or not chunk.text.strip():
                raise ValueError(f"Chunk at index {index} has empty or whitespace-only text")

        vectors = np.asarray(
            self._model.encode(
                [chunk.text for chunk in ordered_chunks],
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        expected_shape = (len(ordered_chunks), self.dimension)
        if vectors.shape != expected_shape:
            raise ValueError(
                f"Model '{self.model_name}' returned shape {vectors.shape}; "
                f"expected {expected_shape} from its reported dimension"
            )

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.isfinite(vectors).all() or np.any(norms == 0):
            raise ValueError("Model returned non-finite or zero-length embedding vectors")
        vectors /= norms
        return EmbeddingBatch(embeddings=vectors, chunks=ordered_chunks)
