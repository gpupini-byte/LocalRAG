# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import numpy as np
import pytest

from localrag.dashboard_services import DashboardConfig, DashboardService
from localrag.embeddings import EmbeddingBatch
from localrag.generator import OllamaUnavailableError
from localrag.retrieval import FaissVectorStore


class FakeEmbedder:
    model_name = "fake-model"
    dimension = 2

    def embed(self, chunks):
        vectors = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(chunks), 1))
        return EmbeddingBatch(vectors, tuple(chunks))


class FakeGenerator:
    def generate(self, messages):
        return "The local index stores vectors for semantic search. [S1]"


def make_service(tmp_path, *, embedder=None, generator_factory=None, max_upload_mb=1):
    config = DashboardConfig(
        documents_dir=tmp_path / "documents",
        index_dir=tmp_path / "index",
        metadata_path=tmp_path / "bibliography" / "records.json",
        embedding_model="fake-model",
        max_upload_mb=max_upload_mb,
    )
    return DashboardService(config, embedder, generator_factory)


def test_empty_index_summary_and_missing_index_error(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder())
    summary = service.index_summary()
    assert summary["exists"] is False
    assert summary["documents"] == summary["chunks"] == 0
    with pytest.raises(FileNotFoundError, match="No FAISS index"):
        service.answer("question")


@pytest.mark.parametrize("name", ["../escape.txt", "sub\\escape.txt", "bad.pdf.exe"])
def test_upload_rejects_unsafe_or_unsupported_filename(tmp_path, name):
    with pytest.raises(ValueError):
        make_service(tmp_path).save_upload(name, b"text")


def test_upload_rejects_oversized_invalid_pdf_and_overwrite(tmp_path):
    service = make_service(tmp_path, max_upload_mb=1)
    with pytest.raises(ValueError, match="exceeds"):
        service.save_upload("large.txt", b"x" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="Invalid PDF"):
        service.save_upload("invalid.pdf", b"not a pdf")
    saved = service.save_upload("notes.txt", b"first")
    assert saved.read_bytes() == b"first"
    with pytest.raises(FileExistsError, match="not overwritten"):
        service.save_upload("notes.txt", b"second")
    assert saved.read_bytes() == b"first"


def test_rebuild_index_ingests_once_and_search_reloads_current_generation(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder())
    service.config.documents_dir.mkdir(parents=True)
    service.save_upload("paper.txt", b"This paper explains vector search and retrieval.")
    progress = []
    result = service.rebuild_index(lambda amount, message: progress.append((amount, message)))
    assert result["documents"] == 1
    assert result["chunks"] >= 1
    assert progress[-1][0] == 1.0
    summary = service.index_summary()
    assert summary["exists"] is True
    assert summary["documents"] == 1
    assert summary["chunks"] == result["chunks"]
    assert summary["model"] == "fake-model"

    # The service loads the active persisted generation for queries, so a newly
    # written generation is visible without sharing a stale in-memory FAISS index.
    store = FaissVectorStore.load(service.config.index_dir)
    assert len(store) == result["chunks"]

    service.save_upload("second.txt", b"A second source about local vector retrieval.")
    second_result = service.rebuild_index()
    refreshed = FaissVectorStore.load(service.config.index_dir)
    assert second_result["documents"] == 2
    assert len(refreshed) == second_result["chunks"]
    assert {Path(chunk.source).name for chunk in refreshed.chunks} == {"paper.txt", "second.txt"}
    assert service.index_summary()["documents"] == 2


def test_empty_document_directory_does_not_write_index(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder())
    service.config.documents_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="No PDF or TXT"):
        service.rebuild_index()
    assert not (service.config.index_dir / "manifest.json").exists()


def test_changed_local_paths_are_used_by_dashboard_service(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder())
    alternate_documents = tmp_path / "alternate-library"
    alternate_index = tmp_path / "alternate-index"
    alternate_documents.mkdir()
    (alternate_documents / "local.txt").write_text("A locally configured research source.", encoding="utf-8")
    service.config = DashboardConfig(
        documents_dir=alternate_documents,
        index_dir=alternate_index,
        metadata_path=tmp_path / "alternate-bibliography" / "records.json",
        embedding_model="fake-model",
    )
    assert [item["name"] for item in service.documents()] == ["local.txt"]
    result = service.rebuild_index()
    assert result["documents"] == 1
    assert (alternate_index / "manifest.json").is_file()


def test_preview_download_delete_and_stale_index_blocking(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder(), generator_factory=FakeGenerator)
    service.config.documents_dir.mkdir(parents=True)
    path = service.save_upload("paper.txt", b"A short, readable research passage.")
    service.rebuild_index()
    assert "readable research passage" in service.document_preview(str(path))
    name, data = service.document_download(str(path))
    assert name == "paper.txt" and data.startswith(b"A short")
    assert service.index_is_stale() is False
    service.delete_document(str(path))
    assert service.index_is_stale() is True
    with pytest.raises(RuntimeError, match="changed since indexing"):
        service.answer("a question")


def test_document_management_rejects_paths_outside_configured_directory(tmp_path):
    service = make_service(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")
    for operation in (service.document_preview, service.document_download, service.delete_document):
        with pytest.raises(ValueError, match="restricted"):
            operation(str(outside))
    assert outside.read_text(encoding="utf-8") == "private"


def test_document_inventory_marks_pending_and_indexed_sources(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder())
    service.config.documents_dir.mkdir(parents=True)
    path = service.save_upload("paper.txt", b"A research source for the index.")
    assert service.document_inventory()[0]["status"] == "Pending indexing"
    service.rebuild_index()
    assert service.document_inventory()[0]["status"] == "Indexed"
    path.write_text("A changed research source for the index.", encoding="utf-8")
    assert service.document_inventory()[0]["status"] == "Changed · rebuild required"


def test_index_time_configuration_change_blocks_queries_until_explicit_rebuild(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder(), generator_factory=FakeGenerator)
    service.config.documents_dir.mkdir(parents=True)
    service.save_upload("paper.txt", b"A source for chunk-size consistency.")
    service.rebuild_index()
    original = service.config
    service.config = DashboardConfig(
        documents_dir=original.documents_dir, index_dir=original.index_dir,
        metadata_path=original.metadata_path, embedding_model=original.embedding_model,
        chunk_size=100, chunk_overlap=20,
    )
    assert service.index_is_stale()
    with pytest.raises(RuntimeError, match="changed since indexing"):
        service.answer("question")


def test_bibliography_refresh_adds_missing_sources_without_crossref(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder())
    service.config.documents_dir.mkdir(parents=True)
    service.save_upload("local-notes.txt", b"Offline notes on research methods.")
    service.rebuild_index()
    assert len(service.bibliography_gaps()) == 1
    first = service.refresh_bibliography()
    second = service.refresh_bibliography()
    assert first == {"added": 1, "records": 1}
    assert second == {"added": 0, "records": 1}
    assert service.bibliography_gaps() == []


def test_ollama_status_reports_only_models_returned_by_local_server(tmp_path, monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return None
        def read(self): return b'{"models":[{"name":"llama3.2:3b"},{"name":"qwen2.5:7b"}]}'

    monkeypatch.setattr("localrag.dashboard_services.urllib.request.urlopen", lambda *args, **kwargs: Response())
    status = make_service(tmp_path).ollama_status()
    assert status["available"] is True
    assert status["models"] == ["llama3.2:3b", "qwen2.5:7b"]


def test_ollama_status_handles_local_server_offline(tmp_path, monkeypatch):
    from urllib.error import URLError
    monkeypatch.setattr("localrag.dashboard_services.urllib.request.urlopen",
                        lambda *args, **kwargs: (_ for _ in ()).throw(URLError("offline")))
    status = make_service(tmp_path).ollama_status()
    assert status["available"] is False
    assert status["models"] == []
    assert "offline" in status["error"]


def test_answer_uses_rebuilt_index_and_propagates_ollama_unavailability(tmp_path):
    service = make_service(tmp_path, embedder=FakeEmbedder(), generator_factory=FakeGenerator)
    service.config.documents_dir.mkdir(parents=True)
    service.save_upload("paper.txt", b"LocalRAG uses FAISS to retrieve semantically similar vectors.")
    service.rebuild_index()
    response = service.answer("What does FAISS do?")
    assert response.status == "ok"
    assert response.citations == ("S1",)
    assert response.evidence[0].source.endswith("paper.txt")

    unavailable = make_service(
        tmp_path,
        embedder=FakeEmbedder(),
        generator_factory=lambda: _UnavailableGenerator(),
    )
    with pytest.raises(OllamaUnavailableError, match="unavailable"):
        unavailable.answer("What does FAISS do?")


class _UnavailableGenerator:
    def generate(self, messages):
        raise OllamaUnavailableError("Ollama unavailable")
