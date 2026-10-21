# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in checks against the user's local index and Ollama installation."""

import os
from pathlib import Path

import pytest

from localrag.dashboard_settings import load_settings
from localrag.embeddings import SentenceTransformerEmbedder
from localrag.generator import OllamaChatClient
from localrag.pipeline import LocalRAGPipeline
from localrag.retrieval import FaissVectorStore


_settings = load_settings()
pytestmark = pytest.mark.skipif(
    os.environ.get("LOCALRAG_RUN_CORPUS_INTEGRATION") != "1"
    or not (Path(_settings.index_dir) / "manifest.json").is_file(),
    reason="set LOCALRAG_RUN_CORPUS_INTEGRATION=1 with the real index, local model cache, and Ollama",
)


def test_real_cpu_questions_against_local_corpus():
    store = FaissVectorStore.load(_settings.index_dir)
    embedder = SentenceTransformerEmbedder(_settings.index.embedding_model, local_files_only=True)
    pipeline = LocalRAGPipeline(
        embedder, store,
        OllamaChatClient(_settings.ollama_model, _settings.ollama_url, _settings.ollama_timeout),
        top_k=_settings.query.top_k,
    )

    direct = pipeline.answer("How do CPU scheduling algorithms improve CPU utilization?")
    assert direct.status in {"ok", "missing_citations", "invalid_citations"}
    assert set(direct.citations) <= {item.source_id for item in direct.evidence}
    assert {Path(item.source).name for item in direct.evidence} >= {
        "Synthetic CPU Scheduling Survey.pdf", "Synthetic Scheduling Article.pdf"
    }

    broad = pipeline.answer("What do these papers say about optimizing CPU?")
    assert broad.status in {"ok", "missing_citations", "invalid_citations"}
    assert set(broad.citations) <= {item.source_id for item in broad.evidence}
    assert all("gpu" not in item.text.casefold() or "cpu" in item.text.casefold()
               for item in broad.evidence)

    unsupported = pipeline.answer("What is the chemical symbol of element 119?")
    assert unsupported.status == "insufficient_evidence"
    assert unsupported.citations == ()
