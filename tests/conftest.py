# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path

import pymupdf
import pytest


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a tiny two-page PDF fixture using PyMuPDF."""
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "Milestone one PDF fixture.")
    second = document.new_page()
    second.insert_text((72, 72), "Second page for metadata checks.")
    document.save(path)
    document.close()
    return path


@pytest.fixture
def empty_pdf(tmp_path: Path) -> Path:
    """Create a valid PDF containing no pages."""
    path = tmp_path / "empty.pdf"
    document = pymupdf.open()
    document.save(path)
    document.close()
    return path
