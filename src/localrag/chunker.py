# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Word-based, metadata-preserving document chunking."""

from dataclasses import dataclass
from hashlib import sha256

from localrag.loader import DocumentRecord


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """A text chunk and its position and source metadata."""

    text: str
    source: str
    page: int | None
    id: str
    position: int


def chunk_documents(
    records: list[DocumentRecord], chunk_size: int = 200, overlap: int = 40
) -> list[ChunkRecord]:
    """Split document records into overlapping word chunks.

    Positions are zero-based within each input record. Chunk IDs are stable for
    an identical ordered input and configuration.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks: list[ChunkRecord] = []
    step = chunk_size - overlap

    for record_number, record in enumerate(records):
        words = record.text.split()
        for position, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start : start + chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            identity = (
                f"{record_number}\0{record.source}\0{record.page}\0"
                f"{position}\0{chunk_text}"
            )
            chunk_id = sha256(identity.encode("utf-8")).hexdigest()
            chunks.append(
                ChunkRecord(
                    text=chunk_text,
                    source=record.source,
                    page=record.page,
                    id=chunk_id,
                    position=position,
                )
            )
            if start + chunk_size >= len(words):
                break

    return chunks
