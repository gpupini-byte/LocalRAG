# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in smoke test: LOCALRAG_RUN_MODEL_INTEGRATION=1 pytest tests/integration."""

import os

import numpy as np
import pytest

from localrag.chunker import ChunkRecord
from localrag.embeddings import SentenceTransformerEmbedder


pytestmark = pytest.mark.skipif(
    os.environ.get("LOCALRAG_RUN_MODEL_INTEGRATION") != "1",
    reason="set LOCALRAG_RUN_MODEL_INTEGRATION=1 to download/load the real model",
)


def test_real_model_shape_dtype_normalization_and_semantic_similarity() -> None:
    embedder = SentenceTransformerEmbedder()
    chunks = [
        ChunkRecord("A dog is playing in the park.", "test", None, "dog-a", 0),
        ChunkRecord("A puppy runs outside in a grassy field.", "test", None, "dog-b", 1),
        ChunkRecord("The stock market closed higher today.", "test", None, "market", 2),
    ]

    result = embedder.embed(chunks)

    assert result.embeddings.shape == (3, 384)
    assert result.embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result.embeddings, axis=1), 1.0, atol=1e-6)
    assert result.chunks == tuple(chunks)
    dog_similarity = float(result.embeddings[0] @ result.embeddings[1])
    unrelated_similarity = float(result.embeddings[0] @ result.embeddings[2])
    assert dog_similarity > unrelated_similarity
