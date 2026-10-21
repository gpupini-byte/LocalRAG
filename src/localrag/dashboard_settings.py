# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Typed, local persistence for dashboard query and ingestion settings."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class QuerySettings:
    preset: str = "BALANCED (DEFAULT)"
    top_k: int = 5
    minimum_score: float | None = None
    crossref: bool = False
    candidate_chunks: int = 25


@dataclass(frozen=True, slots=True)
class IndexSettings:
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 200
    overlap: int = 40


@dataclass(frozen=True, slots=True)
class DashboardSettings:
    documents_dir: str = "data/documents"
    index_dir: str = "data/index"
    metadata_path: str = "data/bibliography/records.json"
    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/chat"
    ollama_timeout: int = 120
    max_upload_mb: int = 50
    query: QuerySettings = field(default_factory=QuerySettings)
    index: IndexSettings = field(default_factory=IndexSettings)

    def validate(self) -> "DashboardSettings":
        if self.query.preset not in {"FAST SEARCH", "BALANCED (DEFAULT)", "DEEP RESEARCH", "CUSTOM"}:
            raise ValueError("Unknown dashboard preset")
        if not 1 <= self.query.top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if self.query.candidate_chunks < self.query.top_k:
            raise ValueError("candidate_chunks must be at least top_k")
        if self.query.minimum_score is not None and not -1 <= self.query.minimum_score <= 1:
            raise ValueError("minimum_score must be between -1 and 1")
        if not 1 <= self.ollama_timeout <= 900 or not 1 <= self.max_upload_mb <= 500:
            raise ValueError("Ollama timeout or upload limit is outside the supported range")
        if self.index.chunk_size <= 0 or not 0 <= self.index.overlap < self.index.chunk_size:
            raise ValueError("Chunk settings must satisfy size > 0 and 0 <= overlap < size")
        if not self.index.embedding_model.strip() or not self.ollama_model.strip() or not self.ollama_url.strip():
            raise ValueError("Model names and Ollama API URL must not be empty")
        parsed = urlparse(self.ollama_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Ollama must use a local loopback URL; remote API hosts are not supported")
        return self


DEFAULT_SETTINGS = DashboardSettings()

PRESETS = {
    "FAST SEARCH": {"top_k": 3, "ollama_timeout": 60, "candidate_chunks": 15},
    "BALANCED (DEFAULT)": {"top_k": 5, "ollama_timeout": 120, "candidate_chunks": 25},
    "DEEP RESEARCH": {"top_k": 8, "ollama_timeout": 180, "candidate_chunks": 60},
}


def load_settings(path: str | Path = "data/dashboard-settings.json") -> DashboardSettings:
    location = Path(path)
    if not location.is_file():
        return DEFAULT_SETTINGS
    try:
        raw = json.loads(location.read_text(encoding="utf-8"))
        result = DashboardSettings(
            documents_dir=raw.get("documents_dir", DEFAULT_SETTINGS.documents_dir),
            index_dir=raw.get("index_dir", DEFAULT_SETTINGS.index_dir),
            metadata_path=raw.get("metadata_path", DEFAULT_SETTINGS.metadata_path),
            ollama_model=raw.get("ollama_model", DEFAULT_SETTINGS.ollama_model),
            ollama_url=raw.get("ollama_url", DEFAULT_SETTINGS.ollama_url),
            ollama_timeout=raw.get("ollama_timeout", DEFAULT_SETTINGS.ollama_timeout),
            max_upload_mb=raw.get("max_upload_mb", DEFAULT_SETTINGS.max_upload_mb),
            query=QuerySettings(**raw.get("query", {})),
            index=IndexSettings(**raw.get("index", {})),
        )
        return result.validate()
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"Dashboard settings file is invalid: {error}") from error


def save_settings(settings: DashboardSettings, path: str | Path = "data/dashboard-settings.json") -> None:
    settings.validate()
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{location.name}.", suffix=".tmp", dir=location.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(asdict(settings), file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, location)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def with_preset(settings: DashboardSettings, preset: str) -> DashboardSettings:
    if preset == "CUSTOM":
        return DashboardSettings(
            documents_dir=settings.documents_dir, index_dir=settings.index_dir,
            metadata_path=settings.metadata_path, ollama_model=settings.ollama_model,
            ollama_url=settings.ollama_url, ollama_timeout=settings.ollama_timeout,
            max_upload_mb=settings.max_upload_mb,
            query=QuerySettings("CUSTOM", settings.query.top_k, settings.query.minimum_score,
                                settings.query.crossref, settings.query.candidate_chunks),
            index=settings.index)
    values = PRESETS[preset]
    return DashboardSettings(
        documents_dir=settings.documents_dir, index_dir=settings.index_dir,
        metadata_path=settings.metadata_path, ollama_model=settings.ollama_model,
        ollama_url=settings.ollama_url, ollama_timeout=values["ollama_timeout"],
        max_upload_mb=settings.max_upload_mb,
        query=QuerySettings(preset, values["top_k"], settings.query.minimum_score,
                            False, values["candidate_chunks"]),
        index=settings.index,
    )


def matches_named_preset(settings: DashboardSettings) -> bool:
    """Whether all preset-controlled values still match the saved named preset.

    Threshold, paths, model names, and index parameters are intentionally not
    part of a query preset, so changing those does not silently select Custom.
    """
    expected = PRESETS.get(settings.query.preset)
    if expected is None:
        return False
    return (
        settings.query.top_k == expected["top_k"]
        and settings.query.candidate_chunks == expected["candidate_chunks"]
        and settings.query.crossref is False
        and settings.ollama_timeout == expected["ollama_timeout"]
    )
