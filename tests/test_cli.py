# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import numpy as np
import pytest

from localrag.chunker import ChunkRecord
from localrag.cli import UninitializedIndexError, _ingest_documents, _load_index, main
from localrag.embeddings import EmbeddingBatch
from localrag.retrieval import FaissVectorStore


class FakeEmbedder:
    dimension = 3
    model_name = "fake-model"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, chunks: list[ChunkRecord]) -> EmbeddingBatch:
        texts = [chunk.text for chunk in chunks]
        self.calls.append(texts)
        vectors = np.asarray([[1, 0, 0] for _ in chunks], dtype=np.float32)
        return EmbeddingBatch(vectors, tuple(chunks))


class FakeGenerator:
    def generate(self, messages: list[dict[str, str]]) -> str:
        return "LocalRAG uses FAISS for vector search. [S1]"


def test_first_run_ingestion_persists_nonempty_index(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notes.txt").write_text("LocalRAG uses FAISS for vector search.", encoding="utf-8")
    embedder = FakeEmbedder()

    store, sources, chunks = _ingest_documents(
        documents, tmp_path / "index", "fake-model", embedder=embedder
    )

    assert sources == 1
    assert chunks == 1
    assert len(store) == 1
    assert embedder.calls == [["LocalRAG uses FAISS for vector search."]]
    restored = FaissVectorStore.load(tmp_path / "index", expected_model_name="fake-model")
    assert len(restored) == 1
    assert restored.search([1, 0, 0])[0].chunk.source.endswith("notes.txt")


def test_persisted_index_is_reused_without_reembedding_documents(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notes.txt").write_text("stored document evidence", encoding="utf-8")
    ingest_embedder = FakeEmbedder()
    _ingest_documents(documents, tmp_path / "index", "fake-model", embedder=ingest_embedder)

    query_embedder = FakeEmbedder()
    monkeypatch.setattr(
        "localrag.embeddings.SentenceTransformerEmbedder", lambda name: query_embedder
    )
    monkeypatch.setattr("localrag.cli.OllamaChatClient", lambda **kwargs: FakeGenerator())
    result = main(
        [
            "What does it say?",
            "--documents",
            str(documents),
            "--index",
            str(tmp_path / "index"),
            "--embedding-model",
            "fake-model",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["status"] == "ok"
    assert len(payload["evidence"]) == 1
    assert query_embedder.calls == [["What does it say?"]]


def test_empty_document_directory_does_not_create_an_empty_index(tmp_path) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    index = tmp_path / "index"

    with pytest.raises(ValueError, match="No non-empty PDF/TXT text"):
        _ingest_documents(documents, index, "fake-model", embedder=FakeEmbedder())

    assert not (index / "manifest.json").exists()


def test_query_rejects_missing_or_empty_index_with_ingest_guidance(tmp_path) -> None:
    with pytest.raises(UninitializedIndexError, match="Run.*ingest"):
        _load_index(tmp_path / "missing", "fake-model")

    empty_store = FaissVectorStore(dimension=3, model_name="fake-model")
    empty_store.save(tmp_path / "empty-index")
    with pytest.raises(UninitializedIndexError, match="contains no vectors"):
        _load_index(tmp_path / "empty-index", "fake-model")
