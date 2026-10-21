# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""Persistent, reviewable bibliographic records kept separately from FAISS."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from localrag.citations import BibliographicMetadata, CitationCandidate, CitationSearchResult

SCHEMA_VERSION = 1
EDITABLE_FIELDS = {"title", "authors", "year", "container", "journal", "volume", "issue", "pages", "doi", "kind"}
REQUIRED_FIELDS = {"title", "authors", "year", "container"}


def canonical_doi(value: str) -> str:
    doi = value.strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I).strip().lower()
    if not re.fullmatch(r"10\.\d{4,9}/[-._;()/:a-z0-9]+", doi, re.I):
        raise ValueError("DOI must have the form 10.<registrant>/<suffix>")
    return doi


def _canonical_document(document_id: str) -> str:
    return str(Path(document_id).expanduser().resolve(strict=False))


def _json_value(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def _value_signature(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True, slots=True)
class BibliographicRecord:
    key: str
    document_ids: tuple[str, ...]
    sources: tuple[str, ...]
    status: str
    fields: dict[str, object]
    overrides: dict[str, tuple[dict[str, object], ...]]
    candidates: dict[str, tuple[dict[str, object], ...]]
    original_pdf_metadata: dict[str, tuple[str, ...]]

    def effective_metadata(self) -> BibliographicMetadata:
        values = {field: value for field, value in self.fields.items()}
        provenance = {field: next((entry.get("provenance", "PDF") for entry in entries
                                   if entry.get("value") == value and entry.get("provenance") == "Crossref"),
                                  next((entry.get("provenance", "PDF") for entry in entries
                                        if entry.get("value") == value), "PDF"))
                      for field, value in self.fields.items()
                      for entries in [self.candidates.get(field, ())]}
        confidence = {field: float(next((entry.get("confidence", 1.0) for entry in entries
                                         if entry.get("value") == value), 1.0))
                      for field, value in self.fields.items()
                      for entries in [self.candidates.get(field, ())]}
        verified = set()
        for field, history in self.overrides.items():
            if history:
                values[field] = history[-1]["approved_value"]
                provenance[field] = "user-provided"
                confidence[field] = 1.0
                verified.add(field)
        if values.get("title") and provenance.get("title") == "PDF":
            from localrag.citations import is_plausible_title
            if not is_plausible_title(str(values["title"])):
                values.pop("title", None)
                provenance.pop("title", None)
                confidence.pop("title", None)
        for field, source in provenance.items():
            unique_values = {_value_signature(entry.get("value"))
                             for entry in self.candidates.get(field, ())}
            if source == "Crossref" and len(unique_values) == 1:
                verified.add(field)
        return BibliographicMetadata(
            title=values.get("title"), authors=tuple(values.get("authors", ())),
            doi=values.get("doi"), year=values.get("year"), container=values.get("container"),
            volume=values.get("volume"), issue=values.get("issue"), pages=values.get("pages"),
            kind=values.get("kind"), provenance=provenance, confidence=confidence,
            verified_fields=tuple(sorted(verified)),
        )

    @property
    def works_cited(self) -> str | None:
        metadata = self.effective_metadata()
        if self.status not in {"verified", "manually_reviewed"}:
            return None
        from localrag.citations import format_mla9
        try:
            return format_mla9(metadata)
        except ValueError:
            return None


class BibliographicMetadataStore:
    """Versioned JSON catalog; atomic replacement preserves data across re-ingestion."""

    def __init__(self, path: str | Path = "data/bibliography/records.json"):
        self.path = Path(path)
        self._data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "records": {}}
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read bibliographic metadata store '{self.path}': {error}") from error
        if (not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION
                or not isinstance(data.get("records"), dict)):
            raise ValueError("Unsupported or malformed bibliographic metadata store")
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file:
                json.dump(self._data, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @staticmethod
    def _key(document_id: str, doi: str | None) -> str:
        return f"doi:{canonical_doi(doi)}" if doi else f"document:{_canonical_document(document_id)}"

    def observe(self, result: CitationSearchResult) -> BibliographicRecord:
        """Save each field observation and candidate without discarding conflicts."""
        metadata = result.metadata
        key = self._key(result.document_id, metadata.doi)
        records = self._data["records"]
        doc_id = _canonical_document(result.document_id)
        # If a reviewer changed the DOI, later searches for the same source keep
        # updating that durable record rather than forking it back to the old DOI.
        for existing_key, existing in records.items():
            if doc_id in existing.get("document_ids", ()):
                key, record = existing_key, existing
                break
        else:
            record = records.setdefault(key, {
                "document_ids": [], "sources": [], "status": "incomplete", "fields": {},
                "overrides": {}, "candidates": {}, "original_pdf_metadata": {},
            })
        if doc_id not in record["document_ids"]:
            record["document_ids"].append(doc_id)
        if result.source not in record["sources"]:
            record["sources"].append(result.source)
        for name, value in metadata.original_pdf_metadata.items():
            raw_values = record["original_pdf_metadata"].setdefault(name, [])
            if value not in raw_values:
                raw_values.append(value)
        snapshots = [(metadata, metadata.provenance, metadata.confidence)] + [
            (candidate.metadata, candidate.metadata.provenance, candidate.metadata.confidence)
            for candidate in result.candidates
        ]
        tracked = ("title", "authors", "year", "container", "volume", "issue", "pages", "doi", "kind")
        for snapshot, provenance, confidence in snapshots:
            for field in tracked:
                value = getattr(snapshot, field, None)
                if value in (None, (), ""):
                    continue
                entry = {"value": _json_value(value), "provenance": provenance.get(field, "PDF"),
                         "confidence": confidence.get(field, 1.0),
                         "verification_status": "candidate"}
                options = record["candidates"].setdefault(field, [])
                if entry not in options:
                    options.append(entry)
                # Never overwrite a reviewer-approved effective value.
                if field not in record["overrides"]:
                    record["fields"][field] = entry["value"]
        # Keep screened-out text available for audit, but never promote it to the
        # effective fields or treat it as an approval-form default.
        for field, value in metadata.rejected_pdf_candidates.items():
            entry = {"value": value, "provenance": "PDF", "confidence": 0.0,
                     "verification_status": "rejected_suspicious"}
            options = record["candidates"].setdefault(field, [])
            if entry not in options:
                options.append(entry)
        if not record["overrides"]:
            record["status"] = result.status
        elif _required_complete(record):
            record["status"] = "manually_reviewed"
        self._save()
        return self.get(key)

    def list_records(self, status: str | None = None) -> list[BibliographicRecord]:
        items = [self._record(key, value) for key, value in self._data["records"].items()]
        return [item for item in items if status is None or item.status == status]

    def get(self, identifier: str) -> BibliographicRecord:
        if identifier in self._data["records"]:
            return self._record(identifier, self._data["records"][identifier])
        canonical = _canonical_document(identifier)
        for key, raw in self._data["records"].items():
            if canonical in raw.get("document_ids", ()) or identifier in raw.get("sources", ()):
                return self._record(key, raw)
        raise KeyError(f"No bibliographic record for '{identifier}'")

    def _record(self, key: str, raw: dict) -> BibliographicRecord:
        return BibliographicRecord(
            key, tuple(raw["document_ids"]), tuple(raw["sources"]), raw["status"],
            dict(raw["fields"]),
            {name: tuple(entries) for name, entries in raw["overrides"].items()},
            {name: tuple(entries) for name, entries in raw["candidates"].items()},
            {name: tuple(values) for name, values in raw.get("original_pdf_metadata", {}).items()},
        )

    def approve(self, identifier: str, field: str, value: object, source_url: str,
                reviewer: str) -> BibliographicRecord:
        if field not in EDITABLE_FIELDS:
            raise ValueError(f"Field must be one of: {', '.join(sorted(EDITABLE_FIELDS))}")
        field = "container" if field == "journal" else field
        approved = _validate_value(field, value)
        parsed = urlparse(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("verification source must be an absolute http(s) URL")
        if not reviewer.strip():
            raise ValueError("reviewer must not be empty")
        current = self.get(identifier)
        new_key = f"doi:{approved}" if field == "doi" else current.key
        if new_key != current.key and new_key in self._data["records"]:
            raise ValueError(f"A record already exists for {new_key}; resolve the duplicate record first")
        raw = self._data["records"][current.key]
        previous = raw["fields"].get(field)
        candidate_values = [entry["value"] for entry in raw["candidates"].get(field, [])]
        original_values = []
        seen_values = set()
        for candidate_value in ([previous] if previous is not None else []) + candidate_values:
            signature = _value_signature(candidate_value)
            if signature not in seen_values:
                seen_values.add(signature)
                original_values.append(candidate_value)
        override = {"original_value": previous, "original_values": original_values,
                    "approved_value": approved, "verification_source_url": source_url,
                    "approved_at": datetime.now(timezone.utc).isoformat(), "reviewer": reviewer.strip()}
        raw["overrides"].setdefault(field, []).append(override)
        raw["fields"][field] = approved
        raw["status"] = "manually_reviewed"
        if field == "doi":
            if new_key != current.key:
                self._data["records"][new_key] = raw
                del self._data["records"][current.key]
        if _required_complete(raw):
            raw["status"] = "manually_reviewed"
        self._save()
        return self.get(new_key)


def _validate_value(field: str, value: object) -> object:
    if field == "authors":
        if isinstance(value, str):
            value = [value]
        if (not isinstance(value, list) or not value or
                any(not isinstance(author, str) or not author.strip() for author in value)):
            raise ValueError("authors must be a non-empty list of names")
        if len(value) > 100:
            raise ValueError("too many author values")
        return [author.strip() for author in value]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    value = value.strip()
    if len(value) > 1000:
        raise ValueError(f"{field} is too long")
    if field == "doi":
        return canonical_doi(value)
    if field == "kind":
        if value.casefold() not in {"journal", "conference"}:
            raise ValueError("kind must be 'journal' or 'conference'")
        return value.casefold()
    if field == "year" and not re.fullmatch(r"(?:1[5-9]|20|21)\d{2}", value):
        raise ValueError("year must be a four-digit publication year")
    if field in {"volume", "issue"} and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 .()/-]{0,39}", value):
        raise ValueError(f"{field} contains unsupported characters")
    if field == "pages" and not re.fullmatch(r"[A-Za-z0-9]+(?:\s*[-–—]\s*[A-Za-z0-9]+)?", value):
        raise ValueError("pages must be a page range or e-locator")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


def _required_complete(raw: dict) -> bool:
    fields = raw.get("fields", {})
    return REQUIRED_FIELDS.issubset(fields) and all(fields.get(name) for name in REQUIRED_FIELDS)
