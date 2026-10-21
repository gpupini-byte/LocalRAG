# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from localrag.chunker import ChunkRecord, chunk_documents
from localrag.loader import DocumentRecord


def test_empty_input_returns_no_chunks() -> None:
    assert chunk_documents([]) == []


def test_short_document_produces_one_chunk() -> None:
    record = DocumentRecord("one two", "notes.txt", None)

    chunks = chunk_documents([record], chunk_size=3, overlap=1)

    assert len(chunks) == 1
    assert chunks[0].text == "one two"
    assert chunks[0].position == 0


def test_exact_chunk_boundary_does_not_create_extra_chunk() -> None:
    record = DocumentRecord("one two three four", "notes.txt", None)

    chunks = chunk_documents([record], chunk_size=2, overlap=0)

    assert [chunk.text for chunk in chunks] == ["one two", "three four"]
    assert [chunk.position for chunk in chunks] == [0, 1]


def test_chunks_overlap_by_configured_number_of_words() -> None:
    record = DocumentRecord("one two three four five six", "notes.txt", None)

    chunks = chunk_documents([record], chunk_size=4, overlap=2)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
    ]


def test_multiple_pages_are_chunked_separately_and_metadata_is_preserved() -> None:
    records = [
        DocumentRecord("alpha beta gamma", "book.pdf", 2),
        DocumentRecord("delta epsilon", "book.pdf", 3),
    ]

    chunks = chunk_documents(records, chunk_size=2, overlap=0)

    assert [(chunk.text, chunk.page, chunk.source, chunk.position) for chunk in chunks] == [
        ("alpha beta", 2, "book.pdf", 0),
        ("gamma", 2, "book.pdf", 1),
        ("delta epsilon", 3, "book.pdf", 0),
    ]


def test_ids_are_deterministic_and_unique_for_duplicate_records() -> None:
    record = DocumentRecord("same text here", "notes.txt", None)

    first = chunk_documents([record, record], chunk_size=2, overlap=0)
    second = chunk_documents([record, record], chunk_size=2, overlap=0)

    assert first == second
    assert all(isinstance(chunk, ChunkRecord) for chunk in first)
    assert len({chunk.id for chunk in first}) == len(first)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_rejects_nonpositive_chunk_size(chunk_size: int) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([], chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 4])
def test_rejects_invalid_overlap(overlap: int) -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents([], chunk_size=4, overlap=overlap)
