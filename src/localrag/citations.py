# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Offline-first semantic source search and conservative MLA citation helpers."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from localrag.chunker import ChunkRecord
from localrag.retrieval import FaissVectorStore

Provenance = Literal["PDF", "user-provided", "Crossref"]


@dataclass(frozen=True, slots=True)
class BibliographicMetadata:
    title: str | None = None
    authors: tuple[str, ...] = ()
    doi: str | None = None
    year: str | None = None
    container: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    kind: Literal["journal", "conference"] | None = None
    original_page: str | None = None
    provenance: dict[str, Provenance] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    verified_fields: tuple[str, ...] = ()
    original_pdf_metadata: dict[str, str] = field(default_factory=dict)
    pdf_candidates: dict[str, str] = field(default_factory=dict)
    rejected_pdf_candidates: dict[str, str] = field(default_factory=dict)

    @property
    def missing_fields(self) -> tuple[str, ...]:
        required = ("title", "authors", "year", "container")
        return tuple(name for name in required if not getattr(self, name))


@dataclass(frozen=True, slots=True)
class CitationCandidate:
    metadata: BibliographicMetadata
    score: float | None = None


@dataclass(frozen=True, slots=True)
class CitationSearchResult:
    rank: int
    document_id: str
    title: str | None
    score: float
    passage: str
    source: str
    page: int | None
    metadata: BibliographicMetadata
    status: Literal["verified", "incomplete", "ambiguous", "manually_reviewed"]
    candidates: tuple[CitationCandidate, ...] = ()

    @property
    def works_cited(self) -> str | None:
        if self.status not in {"verified", "manually_reviewed"}:
            return None
        try:
            return format_mla9(self.metadata)
        except ValueError:
            return None


class CrossrefClient:
    """Optional Crossref REST client; no network requests occur unless called."""

    def __init__(self, base_url: str = "https://api.crossref.org", timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str, params: dict[str, str] | None = None) -> dict:
        url = self.base_url + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        request = urllib.request.Request(url, headers={"User-Agent": "LocalRAG/0.1 (bibliographic lookup)"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise ValueError("Crossref returned an invalid response")
        return payload["message"]

    def by_doi(self, doi: str) -> BibliographicMetadata:
        clean = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        message = self._get("/works/" + urllib.parse.quote(clean, safe=""))
        return crossref_metadata(message)

    def search(self, query: str, rows: int = 5) -> tuple[CitationCandidate, ...]:
        message = self._get("/works", {"query.bibliographic": query, "rows": str(rows)})
        items = message.get("items", [])
        return tuple(CitationCandidate(crossref_metadata(item), item.get("score")) for item in items if isinstance(item, dict))


def crossref_metadata(item: dict) -> BibliographicMetadata:
    def string(value):
        return value.strip() if isinstance(value, str) and value.strip() else None
    title_value = item.get("title")
    container_value = item.get("container-title")
    title = string(title_value[0]) if isinstance(title_value, list) and title_value else None
    container = string(container_value[0]) if isinstance(container_value, list) and container_value else None
    raw_authors = item.get("author", [])
    authors = tuple(" ".join(part for part in (string(a.get("given")), string(a.get("family"))) if part)
                    for a in raw_authors if isinstance(a, dict)) if isinstance(raw_authors, list) else ()
    dates = item.get("published-print") or item.get("published-online") or item.get("created") or {}
    date_parts = dates.get("date-parts", []) if isinstance(dates, dict) else []
    year = str(date_parts[0][0]) if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0] and isinstance(date_parts[0][0], int) else None
    typ = item.get("type")
    kind = "journal" if typ == "journal-article" else ("conference" if typ == "proceedings-article" else None)
    fields = {"title": title, "authors": authors, "doi": string(item.get("DOI")), "year": year,
              "container": container, "volume": string(item.get("volume")),
              "issue": string(item.get("issue")), "pages": string(item.get("page")), "kind": kind}
    provenance = {key: "Crossref" for key, value in fields.items() if value}
    return BibliographicMetadata(**fields, provenance=provenance,
                                 confidence={key: 1.0 for key in provenance},
                                 verified_fields=tuple(provenance))


def extract_pdf_metadata(source: str) -> BibliographicMetadata:
    """Extract cautious candidates from PDF info and first-page front matter.

    Candidate fields are not verified bibliographic facts. In particular, author is
    retained only from explicit PDF info, never inferred from body text.
    """
    import pymupdf
    path = Path(source)
    if path.suffix.casefold() != ".pdf" or not path.is_file():
        return BibliographicMetadata()
    try:
        with pymupdf.open(path) as doc:
            info = doc.metadata or {}
            front = "\n".join(doc[i].get_text() for i in range(min(2, len(doc))))
    except Exception:
        return BibliographicMetadata()
    title, author = info.get("title"), info.get("author")
    original = {key: value.strip() for key, value in info.items()
                if isinstance(value, str) and value.strip()}
    raw_pdf_title = title.strip() if title and title.strip() else None
    pdf_title = raw_pdf_title if raw_pdf_title and _plausible_title(raw_pdf_title) else None
    extracted_title = _front_page_title(front)
    final_title = pdf_title or extracted_title
    explicit_author = (author.strip(),) if author and author.strip() and author.strip().casefold() not in {"unknown", "anonymous", "omar"} else ()
    front_authors = _front_page_authors(front, extracted_title)
    if not explicit_author and front_authors:
        explicit_author = front_authors
    dois = _extract_dois(front)
    doi = dois[0] if len(dois) == 1 else None
    years = re.findall(r"\b(?:19|20)\d{2}\b", front[:5000])
    year = years[0] if years and len(set(years[:4])) == 1 else None
    provenance = {}
    confidence = {}
    for field_name, value, source_name, certainty in (
        ("title", final_title, "PDF", .65 if extracted_title and not pdf_title else .8),
        ("authors", explicit_author, "PDF", .7 if front_authors else .45),
        ("doi", doi, "PDF", .9), ("year", year, "PDF", .65),
    ):
        if value:
            provenance[field_name] = source_name
            confidence[field_name] = certainty
    rejected = {}
    for candidate in (raw_pdf_title, extracted_title):
        if candidate and not _plausible_title(candidate):
            rejected["title"] = candidate
            break
    return BibliographicMetadata(title=final_title, authors=explicit_author, doi=doi, year=year,
                                 provenance=provenance, confidence=confidence,
                                 original_pdf_metadata=original,
                                 pdf_candidates={key: str(value) for key, value in
                                                 (("title", extracted_title), ("doi", doi), ("year", year),
                                                  ("authors", "; ".join(front_authors))) if value},
                                 rejected_pdf_candidates=rejected)


_DOI_RE = re.compile(r"(?i)(?<![A-Z0-9])10\.\d{4,9}/[^\s<>\"']{1,200}")


def _extract_dois(text: str) -> tuple[str, ...]:
    found = set()
    for raw in _DOI_RE.findall(text):
        candidate = raw.strip().rstrip(".,;:]")
        while candidate.endswith(")") and candidate.count("(") < candidate.count(")"):
            candidate = candidate[:-1]
        if re.fullmatch(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", candidate, re.I):
            found.add(candidate.lower())
    return tuple(sorted(found))


def _placeholder_title(title: str) -> bool:
    value = re.sub(r"\s+", " ", title).strip().casefold()
    defaults = ("untitled", "paper title", "document title", "article title", "insert title",
                "title of paper", "title goes here", "sample title", "template")
    return any(value == token or value.startswith(token + " ") or value.startswith(token + " (" )
               for token in defaults) or "use style:" in value


def _plausible_title(title: str) -> bool:
    """Screen obvious paragraphs and template text; passing is not verification."""
    normalized = re.sub(r"\s+", " ", title).strip()
    words = normalized.split()
    body_openers = ("have revolutionized", "has revolutionized", "this paper ",
                    "this study ", "this article ", "we propose ", "we present ",
                    "we investigate ", "in this paper ", "in this study ")
    if (_placeholder_title(normalized) or len(normalized) > 180 or not 2 <= len(words) <= 15
            or re.search(r"[.!?](?:\s|$)", normalized)
            or normalized.count(";") > 1
            or any(phrase in normalized.casefold() for phrase in body_openers)):
        return False
    return True


def is_plausible_title(title: str) -> bool:
    """Return whether a title candidate avoids obvious template/body-text patterns.

    This is a screening heuristic only and does not establish bibliographic truth.
    """
    return _plausible_title(title)


def _front_page_title(text: str) -> str | None:
    lines = [re.sub(r"\s+", " ", line).strip(" \t\r\n•") for line in text.splitlines()]
    doi_line = next((i for i, line in enumerate(lines[:50]) if _extract_dois(line)), -1)
    if doi_line >= 0:
        candidates = []
        for line in lines[doi_line + 1:doi_line + 9]:
            lower = line.casefold()
            if not line:
                if candidates:
                    break
                continue
            if not _plausible_title(line) or "abstract" in lower:
                if candidates:
                    break
                continue
            if re.fullmatch(r"[A-Z][a-z]+\s+(?:[A-Z]\.\s+)?[A-Z][a-z]+(?:\s+\d+)?", line):
                break
            if "@" in line or "department" in lower or "university" in lower:
                break
            if len(line.split()) >= 3 and not line.isupper() and _plausible_title(line):
                candidates.append(line)
            elif candidates:
                break
        if candidates:
            joined = " ".join(candidates)
            if _plausible_title(joined):
                return joined
    for line in lines[:35]:
        lower = line.casefold()
        if (not _plausible_title(line)
                or "10." in line or "doi" in lower or re.fullmatch(r"[\W\d_]+", line)):
            continue
        # Avoid selecting headers, affiliations and journal running labels.
        if any(token in lower for token in ("abstract", "keywords", "university", "department", "journal of")):
            continue
        if len(line.split()) >= 3 and _plausible_title(line):
            return line
    return None


def _front_page_authors(text: str, title: str | None) -> tuple[str, ...]:
    """Read a front-matter byline near a detected title, excluding body citations."""
    if not title:
        return ()
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    title_words = title.casefold().split()[:5]
    title_at = next((i for i, line in enumerate(lines)
                     if len(title_words) >= 3 and all(word in line.casefold() for word in title_words)), None)
    if title_at is None:
        # Multi-line title: use the DOI-associated title start.
        doi_at = next((i for i, line in enumerate(lines) if _extract_dois(line)), None)
        title_at = doi_at if doi_at is not None else -1
    for line in lines[title_at + 1:title_at + 7]:
        if not line or "@" in line or any(term in line.casefold() for term in ("abstract", "department", "university")):
            if line and ("@" in line or "department" in line.casefold()):
                break
            continue
        if not re.search(r"\d\s*[,;*]|\b[A-Z]\.\s+[A-Z]", line):
            continue
        candidates = []
        for fragment in re.split(r",\s*(?=[A-Z])", line):
            name = re.sub(r"\d+|[*†‡]", "", fragment).strip(" ,;:*")
            name = re.sub(r"\s+", " ", name)
            words = name.split()
            if 2 <= len(words) <= 4 and all(re.match(r"^[A-Z][A-Za-z.'-]*$", word) for word in words):
                candidates.append(name)
        if candidates:
            return tuple(candidates)
    return ()


def format_mla9(metadata: BibliographicMetadata) -> str:
    """Format supported article types, assuming fields have been verified."""
    required_verified = {"title", "authors", "year", "container", "kind"}
    verified = set(metadata.verified_fields)
    optional_unverified = any(value and field_name not in verified for field_name, value in
                              (("doi", metadata.doi), ("volume", metadata.volume),
                               ("issue", metadata.issue), ("pages", metadata.pages)))
    if (metadata.missing_fields or not metadata.kind or not required_verified.issubset(verified)
            or optional_unverified):
        raise ValueError("Insufficient verified metadata for MLA 9 formatting")
    authors = metadata.authors
    def mla_author(name: str) -> str:
        parts = name.rsplit(" ", 1)
        return f"{parts[1]}, {parts[0]}" if len(parts) == 2 else name
    author_text = mla_author(authors[0])
    if len(authors) == 2:
        author_text = f"{mla_author(authors[0])}, and {authors[1]}"
    elif len(authors) > 2:
        author_text = f"{mla_author(authors[0])}, et al."
    title = f'"{metadata.title}."'
    container = f"<i>{metadata.container}</i>"
    details = []
    if metadata.volume:
        details.append(f"vol. {metadata.volume}")
    if metadata.issue:
        details.append(f"no. {metadata.issue}")
    details.append(metadata.year or "")
    if metadata.pages:
        details.append(f"pp. {metadata.pages}")
    result = f"{author_text.rstrip('.')}. {title} {container}, {', '.join(x for x in details if x)}."
    if metadata.doi:
        result += f" https://doi.org/{metadata.doi.removeprefix('https://doi.org/')}"
    return result


def mla_in_text(metadata: BibliographicMetadata, original_page: str | None = None) -> str:
    """Create an in-text citation only with verified author/page information."""
    if metadata.missing_fields or not metadata.authors:
        raise ValueError("Insufficient verified author metadata for in-text citation")
    author = metadata.authors[0].split()[-1]
    page = original_page or metadata.original_page
    return f"({author} {page})" if page else f"({author})"


def _metadata_agrees(local: BibliographicMetadata, remote: BibliographicMetadata,
                     exact_doi: bool = False) -> bool:
    if exact_doi and (not local.doi or not remote.doi or local.doi.casefold() != remote.doi.casefold()):
        return False
    if local.title and remote.title:
        left, right = re.sub(r"\W+", " ", local.title.casefold()).strip(), re.sub(r"\W+", " ", remote.title.casefold()).strip()
        if SequenceMatcher(None, left, right).ratio() < .92:
            return False
    if local.year and remote.year and local.year != remote.year:
        return False
    # A front-matter byline is useful corroboration, but remains a candidate field.
    if local.authors and local.confidence.get("authors", 0) >= .65 and remote.authors:
        local_surnames = {author.split()[-1].casefold() for author in local.authors}
        remote_surnames = {author.split()[-1].casefold() for author in remote.authors}
        overlap = len(local_surnames.intersection(remote_surnames))
        denominator = min(len(local_surnames), len(remote_surnames))
        if (denominator == 0 or overlap / denominator < .5
                or local.authors[0].split()[-1].casefold() != remote.authors[0].split()[-1].casefold()):
            return False
    return True


def search_citations(query: str, embedder, store: FaissVectorStore, top_k: int = 5,
                     candidate_chunks: int | None = None, minimum_score: float | None = None,
                     metadata_by_source: dict[str, BibliographicMetadata] | None = None,
                     crossref: CrossrefClient | None = None) -> list[CitationSearchResult]:
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if minimum_score is not None and not -1 <= minimum_score <= 1:
        raise ValueError("minimum_score must be between -1 and 1")
    if not len(store) or not top_k:
        return []
    if embedder.model_name != store.model_name or embedder.dimension != store.dimension:
        raise ValueError("Embedder identity or dimension does not match the indexed model")
    pool = candidate_chunks if candidate_chunks is not None else max(top_k * 5, 20)
    if pool <= 0:
        raise ValueError("candidate_chunks must be non-negative")
    query_chunk = ChunkRecord(text=query, source="<query>", page=None, id="<query>", position=0)
    query_vector = embedder.embed([query_chunk]).embeddings[0]
    pool = min(pool, len(store))
    best: dict[str, object] = {}
    while pool:
        results = store.search(query_vector, pool)
        best = {}
        for hit in results:
            if minimum_score is not None and hit.score < minimum_score:
                continue
            source = hit.chunk.source
            doc_id = str(Path(source).resolve(strict=False))
            if doc_id not in best or hit.score > best[doc_id].score:
                best[doc_id] = hit
        if len(best) >= top_k or pool == len(store):
            break
        pool = min(len(store), max(pool + 1, pool * 2))
    ordered = sorted(best.items(), key=lambda row: (-row[1].score, row[0]))[:top_k]
    output = []
    for rank, (doc_id, hit) in enumerate(ordered, 1):
        metadata = (metadata_by_source or {}).get(hit.chunk.source) or extract_pdf_metadata(hit.chunk.source)
        status = "incomplete"
        candidates: tuple[CitationCandidate, ...] = ()
        # DOI resolution is first choice; disagreements leave the match for review.
        if metadata.doi and crossref:
            try:
                resolved = crossref.by_doi(metadata.doi)
                if _metadata_agrees(metadata, resolved, exact_doi=True):
                    metadata = replace(resolved, original_pdf_metadata=metadata.original_pdf_metadata,
                                       pdf_candidates=metadata.pdf_candidates)
                    status = "verified" if _citation_ready(resolved) else "incomplete"
                elif resolved.title:
                    candidates = (CitationCandidate(resolved),)
                    status = "ambiguous"
            except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to title search if DOI resolution had no record.
                pass
        if crossref and metadata.title and status != "verified" and not candidates:
            try:
                lookup = " ".join(part for part in (metadata.title, metadata.year) if part)
                candidates = crossref.search(lookup)
                matches = [c for c in candidates if _metadata_agrees(metadata, c.metadata)]
                if len(matches) == 1:
                    metadata = replace(matches[0].metadata,
                                       original_pdf_metadata=metadata.original_pdf_metadata,
                                       pdf_candidates=metadata.pdf_candidates)
                    status = "verified" if _citation_ready(metadata) else "incomplete"
                elif len(matches) > 1 or candidates:
                    status = "ambiguous"
            except (OSError, TimeoutError, urllib.error.URLError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
                pass
        if status not in {"verified", "ambiguous"} and metadata.missing_fields:
            status = "incomplete"
        output.append(CitationSearchResult(rank, doc_id, metadata.title, hit.score, hit.chunk.text,
                                           hit.chunk.source, hit.chunk.page, metadata, status, candidates))
    return output


def _citation_ready(metadata: BibliographicMetadata) -> bool:
    required = {"title", "authors", "year", "container", "kind"}
    return not metadata.missing_fields and bool(metadata.kind) and required.issubset(metadata.verified_fields)
