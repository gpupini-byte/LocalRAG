# SPDX-License-Identifier: AGPL-3.0-or-later
import json

import numpy as np
import pymupdf
import pytest

from localrag.chunker import ChunkRecord
from localrag.citations import (BibliographicMetadata, CrossrefClient, format_mla9,
                                extract_pdf_metadata, mla_in_text, search_citations)
from localrag.embeddings import EmbeddingBatch
from localrag.retrieval import FaissVectorStore


class FakeEmbedder:
    model_name = "fake"
    dimension = 3

    def embed(self, chunks):
        return EmbeddingBatch(np.asarray([[1, 0, 0] for _ in chunks], np.float32), tuple(chunks))


def populated_store(tmp_path):
    source_a, source_b = str(tmp_path / "a.txt"), str(tmp_path / "b.txt")
    store = FaissVectorStore(3, "fake")
    chunks = [ChunkRecord("best passage", source_a, 1, "a1", 0),
              ChunkRecord("second page passage", source_a, 2, "a2", 1),
              ChunkRecord("other source", source_b, None, "b1", 0)]
    vectors = np.asarray([[1, 0, 0], [.9, .1, 0], [.8, .6, 0]], np.float32)
    store.add(EmbeddingBatch(vectors, tuple(chunks)))
    return store, source_a, source_b


def test_semantic_rank_dedup_and_best_passage(tmp_path):
    store, source_a, _ = populated_store(tmp_path)
    result = search_citations("topic", FakeEmbedder(), store, top_k=5)
    assert [row.source for row in result] == [source_a, str(tmp_path / "b.txt")]
    assert result[0].passage == "best passage"
    assert result[0].page == 1
    assert len({row.document_id for row in result}) == len(result)


def test_minimum_score_and_empty_index(tmp_path):
    store, _, _ = populated_store(tmp_path)
    assert len(search_citations("topic", FakeEmbedder(), store, minimum_score=.95)) == 1
    assert search_citations("topic", FakeEmbedder(), FaissVectorStore(3, "fake")) == []


def test_metadata_status_and_mla_verified_formatting():
    complete = BibliographicMetadata(title="A Study", authors=("Ada Lovelace",), doi="10.1/x",
        year="2024", container="Journal of Tests", volume="2", issue="1", pages="10-20",
        kind="journal", original_page="12", provenance={"title": "Crossref"},
        verified_fields=("title", "authors", "year", "container", "kind", "doi", "volume", "issue", "pages"))
    assert not complete.missing_fields
    assert format_mla9(complete) == 'Lovelace, Ada. "A Study." <i>Journal of Tests</i>, vol. 2, no. 1, 2024, pp. 10-20. https://doi.org/10.1/x'
    assert mla_in_text(complete) == "(Lovelace 12)"
    with pytest.raises(ValueError, match="Insufficient"):
        format_mla9(BibliographicMetadata(title="Unknown"))


def test_search_offline_and_user_metadata_is_not_claimed_verified(tmp_path):
    store, source, _ = populated_store(tmp_path)
    metadata = BibliographicMetadata(title="User title", provenance={"title": "user-provided"})
    row = search_citations("topic", FakeEmbedder(), store, metadata_by_source={source: metadata})[0]
    assert row.status == "incomplete"
    assert row.works_cited is None


def test_crossref_doi_is_url_encoded_and_failures_propagate(monkeypatch):
    seen = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"message": {"title": ["x"]}}).encode()
    def fake_urlopen(request, timeout):
        seen.append(request.full_url)
        return Response()
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = CrossrefClient(timeout=.2)
    client.by_doi("10.1000/a b")
    assert "10.1000%2Fa%20b" in seen[0]
    def fail(*args, **kwargs): raise TimeoutError("timed out")
    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(TimeoutError):
        client.by_doi("10.1/x")


def test_pdf_front_matter_doi_year_and_placeholder_title(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Journal of Research\ndoi: 10.9999/localrag.synthetic.2020\n"
                     "Synthetic Scheduling Article\nStrategy and Performance Assessment\n"
                     "Alex A. Example\nAbstract\nReceived in 2020")
    doc.set_metadata({"title": "Paper Title (use style: paper title)", "author": ""})
    doc.save(path)
    doc.close()
    metadata = extract_pdf_metadata(str(path))
    assert metadata.title == "Synthetic Scheduling Article Strategy and Performance Assessment"
    assert metadata.doi == "10.9999/localrag.synthetic.2020"
    assert metadata.year == "2020"
    assert metadata.original_pdf_metadata["title"].startswith("Paper Title")
    assert metadata.authors == ("Alex A. Example",)
    assert metadata.confidence["authors"] == .7
    assert "doi" not in metadata.verified_fields


def test_body_paragraph_is_rejected_as_title_and_retained_only_for_review(tmp_path):
    path = tmp_path / "Optimization Techniques for GPU Programming.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    paragraph = ("Graphics Processing Units (GPUs) have revolutionized the field of high performance "
                 "computing by enabling researchers to process large data sets quickly and efficiently. "
                 "This paper examines optimization techniques for programming modern GPU architectures.")
    page.insert_text((72, 72), paragraph)
    doc.set_metadata({"title": paragraph, "author": ""})
    doc.save(path)
    doc.close()
    metadata = extract_pdf_metadata(str(path))
    assert metadata.title is None
    assert metadata.rejected_pdf_candidates["title"] == paragraph
    assert metadata.original_pdf_metadata["title"] == paragraph


def test_filename_is_not_promoted_to_bibliographic_metadata(tmp_path):
    path = tmp_path / "Optimization Techniques for GPU Programming.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "A long paragraph describing graphics processing units and their use.")
    doc.set_metadata({"title": "", "author": ""})
    doc.save(path)
    doc.close()
    metadata = extract_pdf_metadata(str(path))
    assert metadata.title is None
    assert metadata.verified_fields == ()


class FakeCrossref:
    def __init__(self, doi_record=None, candidates=()):
        self.doi_record = doi_record
        self.candidates = tuple(candidates)
        self.lookups = []

    def by_doi(self, doi):
        self.lookups.append(("doi", doi))
        if self.doi_record is None:
            raise OSError("not found")
        return self.doi_record

    def search(self, query):
        self.lookups.append(("search", query))
        from localrag.citations import CitationCandidate
        return tuple(CitationCandidate(m) for m in self.candidates)


def test_exact_doi_resolution_produces_verified_mla(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "DOI 10.9999/localrag.synthetic.2020\nA Synthetic Study of Computing Systems\n2020")
    doc.save(path)
    doc.close()
    store = FaissVectorStore(3, "fake")
    chunk = ChunkRecord("paper evidence", str(path), 1, "p", 0)
    store.add(EmbeddingBatch(np.asarray([[1, 0, 0]], np.float32), (chunk,)))
    resolved = BibliographicMetadata(title="A Synthetic Study of Computing Systems", authors=("Alex Example",),
        doi="10.9999/localrag.synthetic.2020", year="2020", container="Journal of Research", volume="1",
        pages="2-9", kind="journal", provenance={"title": "Crossref"},
        verified_fields=("title", "authors", "doi", "year", "container", "kind", "volume", "pages"))
    crossref = FakeCrossref(doi_record=resolved)
    result = search_citations("topic", FakeEmbedder(), store, crossref=crossref)[0]
    assert crossref.lookups == [("doi", "10.9999/localrag.synthetic.2020")]
    assert result.status == "verified"
    assert "Example, Alex" in result.works_cited


def test_exact_doi_with_author_disagreement_is_ambiguous(tmp_path):
    path = tmp_path / "paper.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "DOI 10.9999/localrag.synthetic.2020\nA Synthetic Study of Computing Systems\n"
        "Alex Example1, Taylor Example2, Jordan Example3\n2020")
    doc.save(path)
    doc.close()
    store = FaissVectorStore(3, "fake")
    chunk = ChunkRecord("paper evidence", str(path), 1, "p", 0)
    store.add(EmbeddingBatch(np.asarray([[1, 0, 0]], np.float32), (chunk,)))
    conflicting = BibliographicMetadata(title="A Synthetic Study of Computing Systems", authors=("Casey Example", "Taylor Example", "Jordan Example"),
        doi="10.9999/localrag.synthetic.2020", year="2020", container="Journal of Research", kind="journal")
    result = search_citations("topic", FakeEmbedder(), store,
                              crossref=FakeCrossref(doi_record=conflicting))[0]
    assert result.status == "ambiguous"
    assert result.works_cited is None
    assert result.candidates[0].metadata.authors[0] == "Casey Example"


def test_ambiguous_crossref_results_are_not_accepted(tmp_path):
    store, source, _ = populated_store(tmp_path)
    local = BibliographicMetadata(title="A Study", doi="10.1/x", year="2020")
    matches = [BibliographicMetadata(title="A Study", authors=("A One",), year="2020", container="J", kind="journal"),
               BibliographicMetadata(title="A Study", authors=("B Two",), year="2020", container="J", kind="journal")]
    # DOI endpoint misses, then title search returns multiple matching candidates.
    crossref = FakeCrossref(candidates=matches)
    rows = search_citations("topic", FakeEmbedder(), store,
                            metadata_by_source={source: local}, crossref=crossref)
    assert rows[0].status == "ambiguous"
    assert rows[0].works_cited is None
    assert len(rows[0].candidates) == 2


def test_candidate_pool_expands_until_unique_source_count_satisfied(tmp_path, monkeypatch):
    store, _, _ = populated_store(tmp_path)
    requested = []
    original_search = store.search
    def track(vector, top_k=5):
        requested.append(top_k)
        return original_search(vector, top_k)
    monkeypatch.setattr(store, "search", track)
    rows = search_citations("topic", FakeEmbedder(), store, top_k=2, candidate_chunks=1)
    assert len(rows) == 2
    assert requested == [1, 2, 3]
