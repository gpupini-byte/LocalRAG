# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in local Ollama check: LOCALRAG_RUN_OLLAMA_INTEGRATION=1 pytest tests/integration."""

import os

import numpy as np
import pytest

from localrag.chunker import ChunkRecord
from localrag.embeddings import EmbeddingBatch
from localrag.generator import OllamaChatClient
from localrag.pipeline import LocalRAGPipeline
from localrag.retrieval import FaissVectorStore


pytestmark = pytest.mark.skipif(
    os.environ.get("LOCALRAG_RUN_OLLAMA_INTEGRATION") != "1",
    reason="set LOCALRAG_RUN_OLLAMA_INTEGRATION=1 with Ollama running and llama3.2:3b pulled",
)


class FixedEmbedder:
    dimension = 3
    model_name = "integration-test-embedding"

    def embed(self, chunks: list[ChunkRecord]) -> EmbeddingBatch:
        return EmbeddingBatch(
            np.asarray([[1, 0, 0] for _ in chunks], dtype=np.float32), tuple(chunks)
        )


def test_real_ollama_returns_cited_grounded_answer() -> None:
    embedder = FixedEmbedder()
    store = FaissVectorStore(dimension=3, model_name=embedder.model_name)
    evidence = ChunkRecord(
        "The LocalRAG integration test's verification value is 731.",
        "integration-fixture.txt",
        None,
        "integration-evidence",
        0,
    )
    store.add(EmbeddingBatch(np.asarray([[1, 0, 0]], dtype=np.float32), (evidence,)))
    pipeline = LocalRAGPipeline(embedder, store, OllamaChatClient())

    result = pipeline.answer("What is the verification value?")

    assert result.status == "ok"
    assert "S1" in result.citations
    assert "731" in result.answer
