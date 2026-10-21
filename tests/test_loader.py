# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import pymupdf
import pytest

from localrag.loader import DocumentRecord, load_document


def make_pdf(path: Path, pages: list[str]) -> Path:
    document = pymupdf.open()
    for content in pages:
        page = document.new_page()
        if content:
            page.insert_text((72, 72), content)
    document.save(path)
    document.close()
    return path


def test_loads_pdf_text_with_one_based_page_numbers(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "sample.pdf", ["First page", "Second page"])

    assert load_document(path) == [
        DocumentRecord("First page\n", str(path), 1),
        DocumentRecord("Second page\n", str(path), 2),
    ]


def test_skips_empty_pdf_pages_without_changing_page_numbers(tmp_path: Path) -> None:
    path = make_pdf(tmp_path / "pages.pdf", ["First", "", "Third"])

    records = load_document(path)

    assert [record.page for record in records] == [1, 3]
    assert [record.text.strip() for record in records] == ["First", "Third"]


def test_loads_txt_as_single_record_with_no_page(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Plain text\n", encoding="utf-8")

    assert load_document(path) == [DocumentRecord("Plain text\n", str(path), None)]


@pytest.mark.parametrize("contents", ["", " \t\n "])
def test_skips_empty_txt_files(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(contents, encoding="utf-8")

    assert load_document(path) == []


def test_missing_file_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="File not found"):
        load_document(path)


def test_malformed_pdf_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_text("not a PDF", encoding="utf-8")

    with pytest.raises(ValueError, match="Could not read PDF"):
        load_document(path)


def test_unsupported_extension_has_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "document.docx"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)
