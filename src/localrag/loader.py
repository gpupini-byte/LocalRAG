# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Document loading interfaces for PDF and plain-text sources."""

from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """Extracted document text with its source and optional one-based page."""

    text: str
    source: str
    page: int | None


def load_document(path: str | Path) -> list[DocumentRecord]:
    """Load a supported document into text records with source metadata."""
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    extension = file_path.suffix.casefold()

    if extension not in (".pdf", ".txt"):
        raise ValueError(f"Unsupported file type: {extension}")

    records = []

    if extension == ".txt":
        text = file_path.read_text(encoding="utf-8")

        if text.strip():
            records.append(
                DocumentRecord(
                    text=text,
                    source=str(file_path),
                    page=None
                )
            )
    else:
        try:
            with pymupdf.open(file_path) as document:
                for page_number, page in enumerate(document, start=1):
                    text = page.get_text()
                    if text.strip():
                        records.append(
                            DocumentRecord(
                                text=text,
                                source=str(file_path),
                                page=page_number,
                            )
                        )
        except pymupdf.FileDataError as error:
            raise ValueError(f"Could not read PDF file '{file_path}': {error}") from error
    return records
