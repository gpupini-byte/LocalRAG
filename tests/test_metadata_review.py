# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import pytest

from localrag.citations import BibliographicMetadata, CitationCandidate, CitationSearchResult
from localrag.metadata_review import BibliographicMetadataStore


def conflicting_result(source: str) -> CitationSearchResult:
    pdf = BibliographicMetadata(
        title="Synthetic CPU Scheduling Study",
        authors=("Alex Example", "Taylor Example", "Jordan Example"),
        doi="10.9999/localrag.synthetic.2020", year="2020",
        provenance={"title": "PDF", "authors": "PDF", "doi": "PDF", "year": "PDF"},
        confidence={"title": .8, "authors": .7, "doi": .9, "year": .65},
        pdf_candidates={"authors": "Alex Example; Taylor Example; Jordan Example"},
        original_pdf_metadata={"author": "Omar", "title": "Paper Title (use style: paper title)"},
    )
    publisher = BibliographicMetadata(
        title="Synthetic CPU Scheduling Study", authors=("Casey Example",),
        doi="10.9999/localrag.synthetic.2020", year="2020", container="Journal of Applied Science and Technology Trends",
        volume="1", issue="1", pages="48-55", kind="journal",
        provenance={key: "Crossref" for key in ("title", "authors", "doi", "year", "container", "volume", "issue", "pages", "kind")},
        confidence={key: 1.0 for key in ("title", "authors", "doi", "year", "container", "volume", "issue", "pages", "kind")},
        verified_fields=("title", "authors", "doi", "year", "container", "volume", "issue", "pages", "kind"),
    )
    return CitationSearchResult(1, source, pdf.title, .59, "CPU scheduling passage", source, 1,
                                pdf, "ambiguous", (CitationCandidate(publisher),))


def test_conflicts_are_persisted_and_manual_approval_creates_mla(tmp_path):
    path = tmp_path / "records.json"
    store = BibliographicMetadataStore(path)
    result = conflicting_result(str(tmp_path / "CPU.pdf"))
    record = store.observe(result)

    assert record.key == "doi:10.9999/localrag.synthetic.2020"
    assert record.status == "ambiguous"
    assert record.candidates["authors"][0]["value"] == ["Alex Example", "Taylor Example", "Jordan Example"]
    assert record.candidates["authors"][1]["value"] == ["Casey Example"]
    assert record.original_pdf_metadata["author"] == ("Omar",)
    assert "Paper Title (use style: paper title)" in record.original_pdf_metadata["title"]
    assert "authors" not in record.effective_metadata().verified_fields
    assert record.works_cited is None

    reviewed = store.approve(
        result.document_id, "authors",
        ["Alex Example", "Taylor Example", "Jordan Example"],
        "https://doi.org/10.9999/localrag.synthetic.2020", "reviewer@example.test",
    )
    assert reviewed.status == "manually_reviewed"
    assert reviewed.works_cited is not None
    assert "Example, Alex" in reviewed.works_cited
    override = reviewed.overrides["authors"][-1]
    assert {tuple(item) if isinstance(item, list) else item for item in override["original_values"]} == {
        ("Alex Example", "Taylor Example", "Jordan Example"), ("Casey Example",)
    }
    assert override["approved_value"] == ["Alex Example", "Taylor Example", "Jordan Example"]
    assert override["verification_source_url"] == "https://doi.org/10.9999/localrag.synthetic.2020"
    assert override["approved_at"].endswith("+00:00")
    restored = BibliographicMetadataStore(path).get(result.document_id)
    assert restored.works_cited == reviewed.works_cited


def test_approval_validation_and_unresolved_fields_never_emit_mla(tmp_path):
    store = BibliographicMetadataStore(tmp_path / "records.json")
    result = conflicting_result(str(tmp_path / "paper.pdf"))
    store.observe(result)
    with pytest.raises(ValueError, match="absolute http"):
        store.approve(result.document_id, "authors", ["Someone"], "doi:10.38094", "reviewer")
    with pytest.raises(ValueError, match="four-digit"):
        store.approve(result.document_id, "year", "20XX", "https://example.test/paper", "reviewer")
    with pytest.raises(ValueError, match="non-empty list"):
        store.approve(result.document_id, "authors", [], "https://example.test/paper", "reviewer")
    with pytest.raises(ValueError, match="kind must"):
        store.approve(result.document_id, "kind", "unknown", "https://example.test/paper", "reviewer")
    reviewed = store.approve(result.document_id, "year", "2020",
                             "https://example.test/paper", "reviewer")
    assert reviewed.status == "manually_reviewed"
    assert reviewed.works_cited is None


def test_reingestion_does_not_remove_reviewed_metadata(tmp_path):
    import numpy as np
    from localrag.embeddings import EmbeddingBatch
    from localrag.cli import _ingest_documents

    documents = tmp_path / "documents"
    documents.mkdir()
    (documents / "notes.txt").write_text("local content", encoding="utf-8")
    catalog_path = tmp_path / "bibliography" / "records.json"
    catalog = BibliographicMetadataStore(catalog_path)
    result = conflicting_result(str(documents / "paper.pdf"))
    catalog.observe(result)
    catalog.approve(result.document_id, "authors", ["Alex Example", "Taylor Example", "Jordan Example"],
                    "https://doi.org/10.9999/localrag.synthetic.2020", "reviewer")
    before = catalog_path.read_bytes()

    class FakeEmbedder:
        dimension = 3
        model_name = "fake-model"
        def embed(self, chunks):
            vectors = np.asarray([[1, 0, 0] for _ in chunks], dtype=np.float32)
            return EmbeddingBatch(vectors, tuple(chunks))

    _ingest_documents(documents, tmp_path / "index", "fake-model", embedder=FakeEmbedder())
    assert catalog_path.read_bytes() == before
    reloaded = BibliographicMetadataStore(catalog_path).get(result.document_id)
    assert reloaded.status == "manually_reviewed"
    assert reloaded.works_cited


def test_doi_override_rekeys_record_without_losing_document_identity(tmp_path):
    store = BibliographicMetadataStore(tmp_path / "records.json")
    result = conflicting_result(str(tmp_path / "paper.pdf"))
    store.observe(result)
    updated = store.approve(result.document_id, "doi", "10.9999/corrected", "https://example.test/record", "reviewer")
    assert updated.key == "doi:10.9999/corrected"
    assert store.get(result.document_id).key == updated.key
    assert [item.key for item in store.list_records()] == [updated.key]


def test_cli_inspection_approval_and_mla(tmp_path, capsys):
    from localrag.cli import main
    catalog = BibliographicMetadataStore(tmp_path / "records.json")
    result = conflicting_result(str(tmp_path / "paper.pdf"))
    catalog.observe(result)
    assert main(["citation-review", "list", "--status", "ambiguous", "--store", str(catalog.path)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing[0]["status"] == "ambiguous"
    assert main(["citation-review", "approve", result.document_id, "--store", str(catalog.path),
                 "--field", "authors", "--value-json",
                 '["Alex Example", "Taylor Example", "Jordan Example"]',
                 "--source-url", "https://doi.org/10.9999/localrag.synthetic.2020", "--reviewer", "test-reviewer"]) == 0
    approved = json.loads(capsys.readouterr().out)
    assert approved["status"] == "manually_reviewed"
    assert "Example, Alex" in approved["works_cited"]
    assert main(["citation-review", "mla", result.document_id, "--store", str(catalog.path)]) == 0
    assert json.loads(capsys.readouterr().out)["works_cited"] == approved["works_cited"]


def test_search_cli_reuses_saved_author_override(tmp_path, monkeypatch, capsys):
    import localrag.citations as citations
    import localrag.cli as cli

    catalog = BibliographicMetadataStore(tmp_path / "records.json")
    result = conflicting_result(str(tmp_path / "paper.pdf"))
    catalog.observe(result)
    catalog.approve(result.document_id, "authors",
                    ["Alex Example", "Taylor Example", "Jordan Example"],
                    "https://doi.org/10.9999/localrag.synthetic.2020", "reviewer")
    monkeypatch.setattr(cli, "_load_index", lambda *_: object())
    monkeypatch.setattr("localrag.embeddings.SentenceTransformerEmbedder", lambda *_: object())
    monkeypatch.setattr(citations, "search_citations", lambda *args, **kwargs: [result])

    assert cli.main(["search-citations", "CPU management", "--index", "unused",
                     "--embedding-model", "fake", "--metadata-store", str(catalog.path)]) == 0
    payload = json.loads(capsys.readouterr().out)["results"][0]
    assert payload["metadata_status"] == "manually_reviewed"
    assert "Example, Alex" in payload["mla_works_cited"]


def test_suspicious_pdf_title_remains_a_rejected_candidate_not_effective_metadata(tmp_path):
    suspicious = "Graphics Processing Units have revolutionized computing. This article studies GPUs."
    source = str(tmp_path / "GPU Optimization.pdf")
    metadata = BibliographicMetadata(rejected_pdf_candidates={"title": suspicious},
                                    original_pdf_metadata={"title": suspicious})
    result = CitationSearchResult(1, source, None, .5, "passage", source, 1, metadata, "incomplete")
    record = BibliographicMetadataStore(tmp_path / "records.json").observe(result)
    assert record.effective_metadata().title is None
    assert record.candidates["title"][0]["verification_status"] == "rejected_suspicious"
