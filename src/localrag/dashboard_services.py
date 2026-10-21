# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""UI-independent service layer used by the Streamlit dashboard."""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable
from urllib.parse import urlparse

from localrag.chunker import chunk_documents
from localrag.citations import CrossrefClient, CitationSearchResult, search_citations
from localrag.citations import extract_pdf_metadata
from localrag.embeddings import EmbeddingBatch, SentenceTransformerEmbedder
from localrag.generator import DEFAULT_GENERATION_MODEL, DEFAULT_OLLAMA_URL, OllamaChatClient
from localrag.loader import load_document
from localrag.metadata_review import BibliographicMetadataStore
from localrag.pipeline import GroundedResponse, LocalRAGPipeline
from localrag.retrieval import FaissVectorStore


_INDEX_LOCKS: dict[str, threading.Lock] = {}
_INDEX_LOCKS_GUARD = threading.Lock()


def _index_lock(directory: Path) -> threading.Lock:
    key = str(directory.resolve(strict=False))
    with _INDEX_LOCKS_GUARD:
        return _INDEX_LOCKS.setdefault(key, threading.Lock())


@dataclass(frozen=True, slots=True)
class DashboardConfig:
    documents_dir: Path = Path("data/documents")
    index_dir: Path = Path("data/index")
    metadata_path: Path = Path("data/bibliography/records.json")
    embedding_model: str = "all-MiniLM-L6-v2"
    ollama_model: str = DEFAULT_GENERATION_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    ollama_timeout: float = 120.0
    max_upload_mb: int = 50
    chunk_size: int = 200
    chunk_overlap: int = 40

    def __post_init__(self) -> None:
        if not 1 <= self.max_upload_mb <= 500:
            raise ValueError("max_upload_mb must be between 1 and 500")
        if self.chunk_size <= 0 or not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk settings must satisfy size > 0 and 0 <= overlap < size")
        parsed = urlparse(self.ollama_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama API URL must use a local loopback host")


class LockedEmbedder:
    """Serialize access to a cached SentenceTransformer across Streamlit sessions."""

    def __init__(self, embedder: SentenceTransformerEmbedder):
        self._embedder = embedder
        self._lock = threading.RLock()
        self.model_name = embedder.model_name
        self.dimension = embedder.dimension

    def embed(self, chunks) -> EmbeddingBatch:
        with self._lock:
            return self._embedder.embed(chunks)


class DashboardService:
    def __init__(self, config: DashboardConfig, embedder=None,
                 generator_factory: Callable[[], object] | None = None):
        self.config = config
        self.embedder = embedder
        self.generator_factory = generator_factory or (
            lambda: OllamaChatClient(config.ollama_model, config.ollama_url, config.ollama_timeout)
        )

    def _embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(self.config.embedding_model, local_files_only=True)
        return self.embedder

    def index_summary(self) -> dict:
        manifest = self.config.index_dir / "manifest.json"
        if not manifest.is_file():
            return {"exists": False, "documents": 0, "chunks": 0,
                    "model": None, "dimension": None, "error": "No index exists. Add documents and rebuild the index."}
        try:
            store = FaissVectorStore.load(self.config.index_dir)
        except (OSError, ValueError) as error:
            return {"exists": False, "documents": 0, "chunks": 0,
                    "model": None, "dimension": None, "error": f"Index could not be loaded: {error}"}
        sources = {chunk.source for chunk in store.chunks}
        stale = self.index_is_stale(store)
        mismatch = store.model_name != self.config.embedding_model
        error = None
        if mismatch:
            error = (f"This index uses '{store.model_name}' but settings select '{self.config.embedding_model}'. "
                     "Confirm an index rebuild before searching.")
        elif stale:
            error = "Documents or index-time settings changed. Rebuild the index before searching."
        return {"exists": True, "documents": len(sources), "chunks": len(store),
                "model": store.model_name, "dimension": store.dimension, "error": error,
                "stale": stale or mismatch}

    def documents(self) -> list[dict[str, object]]:
        if not self.config.documents_dir.is_dir():
            return []
        return [{"name": path.name, "path": str(path), "size_bytes": path.stat().st_size,
                 "type": path.suffix.casefold().lstrip("."), "modified": path.stat().st_mtime}
                for path in sorted(self.config.documents_dir.rglob("*"))
                if path.is_file() and not path.is_symlink()
                and path.suffix.casefold() in {".pdf", ".txt"}]

    def document_inventory(self) -> list[dict[str, object]]:
        files = self.documents()
        indexed, signatures, saved_index = self._indexed_state()
        for item in files:
            path = Path(item["path"])
            identity = str(path.resolve(strict=False))
            signature = _file_signature(path)
            if identity not in indexed:
                status = "Pending indexing"
            elif signatures is not None and signatures.get(identity) != signature:
                status = "Changed · rebuild required"
            else:
                status = "Indexed"
            item["status"] = status
        return files

    def document_preview(self, source: str, character_limit: int = 2400) -> str:
        path = self._safe_document_path(source)
        records = load_document(path)
        text = "\n\n".join(record.text for record in records)
        return text[:character_limit] + ("…" if len(text) > character_limit else "")

    def document_download(self, source: str) -> tuple[str, bytes]:
        path = self._safe_document_path(source)
        return path.name, path.read_bytes()

    def delete_document(self, source: str) -> Path:
        path = self._safe_document_path(source)
        path.unlink()
        return path

    def save_upload(self, filename: str, content: bytes) -> Path:
        if not isinstance(content, bytes):
            raise ValueError("Uploaded file content must be bytes")
        if not filename or "\x00" in filename or "/" in filename or "\\" in filename or ":" in filename:
            raise ValueError("Filename must be a plain filename without path separators")
        if PurePosixPath(filename).name != filename or PureWindowsPath(filename).name != filename:
            raise ValueError("Path traversal is not allowed")
        suffix = Path(filename).suffix.casefold()
        if suffix not in {".pdf", ".txt"}:
            raise ValueError("Only PDF and TXT documents are supported")
        limit = self.config.max_upload_mb * 1024 * 1024
        if len(content) == 0:
            raise ValueError("Uploaded file is empty")
        if len(content) > limit:
            raise ValueError(f"File exceeds the {self.config.max_upload_mb} MB upload limit")
        if suffix == ".txt":
            try:
                content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError("TXT files must use UTF-8 encoding") from error
        else:
            import pymupdf
            try:
                with pymupdf.open(stream=content, filetype="pdf") as document:
                    if document.is_encrypted:
                        raise ValueError("Encrypted PDFs are not supported")
            except ValueError:
                raise
            except Exception as error:
                raise ValueError(f"Invalid PDF file: {error}") from error

        self.config.documents_dir.mkdir(parents=True, exist_ok=True)
        root = self.config.documents_dir.resolve()
        target = (root / filename).resolve()
        if target.parent != root:
            raise ValueError("Upload target must remain inside the configured documents directory")
        try:
            with target.open("xb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
        except FileExistsError as error:
            raise FileExistsError(f"A document named '{filename}' already exists; it was not overwritten") from error
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def rebuild_index(self, progress: Callable[[float, str], None] | None = None) -> dict:
        with _index_lock(self.config.index_dir):
            return self._rebuild_index_locked(progress)

    def _rebuild_index_locked(self, progress: Callable[[float, str], None] | None = None) -> dict:
        if not self.config.documents_dir.is_dir():
            raise FileNotFoundError(f"Documents directory not found: {self.config.documents_dir}")
        paths = [path for path in sorted(self.config.documents_dir.rglob("*"))
                 if path.is_file() and not path.is_symlink()
                 and path.suffix.casefold() in {".pdf", ".txt"}]
        if not paths:
            raise ValueError("No PDF or TXT documents found; no index was written")
        records = []
        for index, path in enumerate(paths, start=1):
            records.extend(load_document(path))
            if progress:
                progress(.4 * index / len(paths), f"Loaded {index} of {len(paths)} documents")
        chunks = chunk_documents(records, self.config.chunk_size, self.config.chunk_overlap)
        if not chunks:
            raise ValueError("No non-empty text was found; no index was written")
        if progress:
            progress(.45, f"Created {len(chunks)} chunks")
        embedder = self._embedder()
        if embedder.model_name != self.config.embedding_model:
            raise ValueError("Loaded embedding model does not match the explicitly configured index model")
        batch = embedder.embed(chunks)
        if progress:
            progress(.9, "Writing a new persistent FAISS generation")
        vector_store = FaissVectorStore(dimension=embedder.dimension, model_name=embedder.model_name)
        vector_store.add(batch)
        vector_store.save(self.config.index_dir)
        validated = FaissVectorStore.load(self.config.index_dir, expected_model_name=vector_store.model_name)
        if len(validated) != len(vector_store) or validated.dimension != embedder.dimension:
            raise ValueError("The newly written FAISS index did not pass validation")
        self._write_index_state(paths)
        if progress:
            progress(1.0, "Index rebuild complete")
        return {"documents": len(paths), "chunks": len(chunks), "vectors": len(vector_store),
                "embedding_model": vector_store.model_name}

    def answer(self, question: str, top_k: int = 5,
               minimum_score: float | None = None) -> GroundedResponse:
        store = self._load_nonempty_index()
        return LocalRAGPipeline(self._embedder(), store, self.generator_factory(),
                                top_k=top_k, minimum_score=minimum_score).answer(question)

    def search_sources(self, query: str, top_k: int = 5, candidate_chunks: int | None = None,
                       minimum_score: float | None = None, use_crossref: bool = False,
                       crossref_timeout: float = 5.0) -> list[CitationSearchResult]:
        store = self._load_nonempty_index()
        results = search_citations(query, self._embedder(), store, top_k, candidate_chunks,
                                   minimum_score, crossref=(CrossrefClient(timeout=crossref_timeout)
                                                            if use_crossref else None))
        registry = BibliographicMetadataStore(self.config.metadata_path)
        from dataclasses import replace
        return [replace(result, metadata=(record := registry.observe(result)).effective_metadata(),
                        status=record.status) for result in results]

    def review_store(self) -> BibliographicMetadataStore:
        return BibliographicMetadataStore(self.config.metadata_path)

    def refresh_bibliography(self) -> dict[str, int]:
        """Add missing indexed-source records without replacing reviewer approvals."""
        store = self._load_nonempty_index(allow_stale=True)
        registry = BibliographicMetadataStore(self.config.metadata_path)
        known = {document_id for record in registry.list_records() for document_id in record.document_ids}
        sources = sorted({chunk.source for chunk in store.chunks})
        added = 0
        for source in sources:
            if not Path(source).is_file():
                continue
            path_id = str(Path(source).resolve(strict=False))
            if path_id in known:
                continue
            metadata = extract_pdf_metadata(source)
            from localrag.citations import CitationSearchResult
            result = CitationSearchResult(1, path_id, metadata.title, 0.0, "", source, None,
                                         metadata, "incomplete")
            registry.observe(result)
            added += 1
        return {"added": added, "records": len(registry.list_records())}

    def bibliography_gaps(self) -> list[str]:
        if not (self.config.index_dir / "manifest.json").is_file():
            return []
        try:
            store = FaissVectorStore.load(self.config.index_dir)
            known = {document_id for record in self.review_store().list_records()
                     for document_id in record.document_ids}
            return sorted(source for source in {chunk.source for chunk in store.chunks}
                          if str(Path(source).resolve(strict=False)) not in known)
        except (OSError, ValueError):
            return []

    def ollama_status(self, timeout: float = 1.0) -> dict[str, object]:
        base = self.config.ollama_url.rsplit("/api/", 1)[0]
        parsed = urlparse(base)
        if parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            return {"available": False, "models": [], "error": "Ollama must be local to this machine"}
        request = urllib.request.Request(base + "/api/tags", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = [item.get("name") for item in payload.get("models", []) if isinstance(item, dict)]
            return {"available": True, "models": models, "error": None}
        except (OSError, urllib.error.URLError, TimeoutError, socket.timeout, json.JSONDecodeError,
                AttributeError, TypeError) as error:
            return {"available": False, "models": [], "error": str(error)}

    def evaluation_summary(self, report_path: Path = Path("evaluation/results/latest.json")) -> dict | None:
        if not report_path.is_file():
            return None
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            benchmark = payload.get("benchmark", {})
            configurations = payload.get("configurations", [])
            return {"title": benchmark.get("title", "Evaluation report"),
                    "label_type": benchmark.get("label_type", "unknown"),
                    "question_count": benchmark.get("question_count"),
                    "document_count": benchmark.get("document_count"),
                    "configurations": [{"chunk_size": item.get("chunk_size_words"),
                                        "overlap": item.get("overlap_words"),
                                        "recall_at_5": item.get("retrieval_metrics", {}).get("recall_at_k", {}).get("5"),
                                        "mrr": item.get("retrieval_metrics", {}).get("mrr")}
                                       for item in configurations],
                    "source": str(report_path)}
        except (OSError, json.JSONDecodeError, TypeError):
            return None

    def _load_nonempty_index(self, allow_stale: bool = False) -> FaissVectorStore:
        if not (self.config.index_dir / "manifest.json").is_file():
            raise FileNotFoundError(f"No FAISS index at '{self.config.index_dir}'. Rebuild it from Documents first.")
        store = FaissVectorStore.load(self.config.index_dir)
        if not len(store):
            raise ValueError("The FAISS index exists but contains no vectors. Rebuild it after adding documents.")
        if not allow_stale and self.index_is_stale(store):
            raise RuntimeError("The document library has changed since indexing. Rebuild the index before searching.")
        return store

    def index_is_stale(self, store: FaissVectorStore | None = None) -> bool:
        manifest = self.config.index_dir / "manifest.json"
        if not manifest.is_file():
            return False
        store = store or FaissVectorStore.load(self.config.index_dir)
        indexed, signatures, saved_index = self._indexed_state(store)
        current_paths = {str(Path(item["path"]).resolve(strict=False)) for item in self.documents()}
        if current_paths != indexed:
            return True
        if store.model_name != self.config.embedding_model:
            return True
        if saved_index is not None and saved_index != {
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "overlap": self.config.chunk_overlap,
        }:
            return True
        if signatures is not None:
            current_signatures = {str(Path(item["path"]).resolve(strict=False)): _file_signature(Path(item["path"]))
                                  for item in self.documents()}
            return current_signatures != signatures
        return False

    def _indexed_state(self, store: FaissVectorStore | None = None):
        manifest = self.config.index_dir / "manifest.json"
        if not manifest.is_file():
            return set(), None, None
        try:
            store = store or FaissVectorStore.load(self.config.index_dir)
        except (OSError, ValueError):
            return set(), None, None
        indexed = {str(Path(chunk.source).resolve(strict=False)) for chunk in store.chunks}
        state_path = self.config.index_dir / "dashboard-state.json"
        if not state_path.is_file():
            return indexed, None, None
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), dict):
                return indexed, None, {"invalid": True}
            saved_index = payload.get("index")
            if saved_index is not None and not isinstance(saved_index, dict):
                return indexed, None, {"invalid": True}
            return indexed, payload["files"], saved_index
        except (OSError, json.JSONDecodeError, AttributeError):
            return indexed, None, {"invalid": True}

    def _write_index_state(self, paths: list[Path]) -> None:
        state_path = self.config.index_dir / "dashboard-state.json"
        payload = {"schema_version": 1,
                   "index": {"embedding_model": self.config.embedding_model,
                             "chunk_size": self.config.chunk_size,
                             "overlap": self.config.chunk_overlap},
                   "files": {str(path.resolve(strict=False)): _file_signature(path) for path in paths}}
        descriptor, temporary = tempfile.mkstemp(prefix=".dashboard-state.", suffix=".tmp",
                                                  dir=self.config.index_dir)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
                json.dump(payload, file, ensure_ascii=False, separators=(",", ":"))
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, state_path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def _safe_document_path(self, source: str) -> Path:
        root = self.config.documents_dir.resolve()
        target = Path(os.path.abspath(source))
        try:
            relative = target.relative_to(root)
        except ValueError as error:
            raise ValueError("Document operation is restricted to the configured documents folder") from error
        current = root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("Symbolic links cannot be previewed, downloaded, or deleted")
        target = current.resolve(strict=True)
        if target != current:
            raise ValueError("Document operation cannot follow symbolic links")
        if not target.is_file() or target.suffix.casefold() not in {".pdf", ".txt"}:
            raise ValueError("Only existing PDF and TXT documents can be previewed, downloaded, or deleted")
        return target


def _file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "modified_ns": stat.st_mtime_ns}
