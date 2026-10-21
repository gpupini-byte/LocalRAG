# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Guilherme Pupini
"""LocalRAG's local Streamlit research workspace.

Run from the repository root with ``python -m streamlit run streamlit_app.py``.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st

from localrag.dashboard_services import DashboardConfig, DashboardService, LockedEmbedder
from localrag.dashboard_settings import (DEFAULT_SETTINGS, DashboardSettings, IndexSettings,
                                         QuerySettings, load_settings, matches_named_preset,
                                         save_settings, with_preset)
from localrag.citations import is_plausible_title
from localrag.embeddings import SentenceTransformerEmbedder


PAGES = ("Overview", "Ask AI", "Citation Search", "Documents", "Citation Review")


@st.cache_resource(show_spinner="Loading local embedding model…")
def _cached_embedder(model_name: str) -> LockedEmbedder:
    """Cache one model per process; LockedEmbedder guards concurrent use."""
    return LockedEmbedder(SentenceTransformerEmbedder(model_name, local_files_only=True))


def _service(config: DashboardConfig, need_embedder: bool = False) -> DashboardService:
    embedder = _cached_embedder(config.embedding_model) if need_embedder else None
    return DashboardService(config, embedder=embedder)


_SETTINGS_PATH = Path(os.environ.get("LOCALRAG_DASHBOARD_SETTINGS", "data/dashboard-settings.json"))


def _initialize_settings() -> None:
    # Apply deferred cross-screen widget updates before any widgets are created.
    for key, value in st.session_state.pop("_pending_widget_state", {}).items():
        st.session_state[key] = value
    pending_page = st.session_state.pop("_pending_navigation", None)
    if pending_page in PAGES:
        st.session_state["workspace_page"] = pending_page
        st.session_state["workspace_page_widget"] = pending_page
    settings = load_settings(_SETTINGS_PATH)
    defaults = {
        "preset": settings.query.preset,
        "query_top_k": settings.query.top_k,
        "ask_top_k": settings.query.top_k,
        "search_top_k": settings.query.top_k,
        "minimum_enabled": settings.query.minimum_score is not None,
        "minimum_score": settings.query.minimum_score if settings.query.minimum_score is not None else 0.25,
        "crossref": settings.query.crossref,
        "candidate_chunks": settings.query.candidate_chunks,
        "search_candidate_chunks": settings.query.candidate_chunks,
        "ask_filter_minimum": settings.query.minimum_score is not None,
        "search_filter_minimum": settings.query.minimum_score is not None,
        "ask_minimum_value": settings.query.minimum_score if settings.query.minimum_score is not None else 0.25,
        "search_minimum_value": settings.query.minimum_score if settings.query.minimum_score is not None else 0.25,
        "search_crossref": settings.query.crossref,
        "config_documents": settings.documents_dir,
        "config_index": settings.index_dir,
        "config_metadata": settings.metadata_path,
        "config_embedding_model": settings.index.embedding_model,
        "config_chunk_size": settings.index.chunk_size,
        "config_chunk_overlap": settings.index.overlap,
        "config_ollama_model": settings.ollama_model,
        "config_ollama_url": settings.ollama_url,
        "config_ollama_timeout": settings.ollama_timeout,
        "config_upload_limit": settings.max_upload_mb,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _settings_from_state(preset: str | None = None) -> DashboardSettings:
    selected_preset = preset or st.session_state.get("preset", "CUSTOM")
    top_k = int(st.session_state.get("query_top_k", 5))
    settings = DashboardSettings(
        documents_dir=st.session_state.get("config_documents", "data/documents"),
        index_dir=st.session_state.get("config_index", "data/index"),
        metadata_path=st.session_state.get("config_metadata", "data/bibliography/records.json"),
        ollama_model=st.session_state.get("config_ollama_model", "llama3.2:3b"),
        ollama_url=st.session_state.get("config_ollama_url", "http://localhost:11434/api/chat"),
        ollama_timeout=int(st.session_state.get("config_ollama_timeout", 120)),
        max_upload_mb=int(st.session_state.get("config_upload_limit", 50)),
        query=QuerySettings(
            selected_preset, top_k,
            float(st.session_state.get("minimum_score", .25)) if st.session_state.get("minimum_enabled") else None,
            bool(st.session_state.get("crossref", False)),
            max(top_k, int(st.session_state.get("candidate_chunks", 25)))),
        index=IndexSettings(
            st.session_state.get("config_embedding_model", "all-MiniLM-L6-v2"),
            int(st.session_state.get("config_chunk_size", 200)),
            int(st.session_state.get("config_chunk_overlap", 40))),
    )
    return settings.validate()


def _save_settings_callback() -> None:
    try:
        if st.session_state.config_chunk_overlap >= st.session_state.config_chunk_size:
            st.session_state.config_chunk_overlap = st.session_state.config_chunk_size - 1
            st.session_state.settings_save_notice = "Overlap was reduced to remain smaller than chunk size."
        st.session_state.ask_top_k = st.session_state.query_top_k
        st.session_state.search_top_k = st.session_state.query_top_k
        st.session_state.search_candidate_chunks = st.session_state.candidate_chunks
        st.session_state.ask_filter_minimum = st.session_state.minimum_enabled
        st.session_state.search_filter_minimum = st.session_state.minimum_enabled
        st.session_state.ask_minimum_value = st.session_state.minimum_score
        st.session_state.search_minimum_value = st.session_state.minimum_score
        st.session_state.search_crossref = st.session_state.crossref
        settings = _settings_from_state()
        if settings.query.preset != "CUSTOM" and not matches_named_preset(settings):
            st.session_state.preset = "CUSTOM"
            settings = _settings_from_state()
        save_settings(settings, _SETTINGS_PATH)
        st.session_state.pop("settings_save_error", None)
    except (ValueError, OSError) as error:
        st.session_state.settings_save_error = str(error)


def _preset_changed() -> None:
    current = _settings_from_state()
    preset = st.session_state.preset
    updated = with_preset(current, preset)
    st.session_state.query_top_k = updated.query.top_k
    st.session_state.ask_top_k = updated.query.top_k
    st.session_state.search_top_k = updated.query.top_k
    st.session_state.candidate_chunks = updated.query.candidate_chunks
    st.session_state.search_candidate_chunks = updated.query.candidate_chunks
    st.session_state.crossref = updated.query.crossref
    st.session_state.search_crossref = updated.query.crossref
    st.session_state.config_ollama_timeout = updated.ollama_timeout
    save_settings(updated, _SETTINGS_PATH)


def _installed_model_changed() -> None:
    selected = st.session_state.get("installed_ollama_model")
    if selected:
        st.session_state.config_ollama_model = selected
        _save_settings_callback()


def _reset_settings() -> None:
    for key, value in {
        "preset": DEFAULT_SETTINGS.query.preset,
        "query_top_k": DEFAULT_SETTINGS.query.top_k,
        "ask_top_k": DEFAULT_SETTINGS.query.top_k,
        "search_top_k": DEFAULT_SETTINGS.query.top_k,
        "minimum_enabled": False,
        "minimum_score": 0.25,
        "crossref": False,
        "candidate_chunks": DEFAULT_SETTINGS.query.candidate_chunks,
        "search_candidate_chunks": DEFAULT_SETTINGS.query.candidate_chunks,
        "ask_filter_minimum": False,
        "search_filter_minimum": False,
        "ask_minimum_value": 0.25,
        "search_minimum_value": 0.25,
        "search_crossref": False,
        "config_documents": DEFAULT_SETTINGS.documents_dir,
        "config_index": DEFAULT_SETTINGS.index_dir,
        "config_metadata": DEFAULT_SETTINGS.metadata_path,
        "config_embedding_model": DEFAULT_SETTINGS.index.embedding_model,
        "config_chunk_size": DEFAULT_SETTINGS.index.chunk_size,
        "config_chunk_overlap": DEFAULT_SETTINGS.index.overlap,
        "config_ollama_model": DEFAULT_SETTINGS.ollama_model,
        "config_ollama_url": DEFAULT_SETTINGS.ollama_url,
        "config_ollama_timeout": DEFAULT_SETTINGS.ollama_timeout,
        "config_upload_limit": DEFAULT_SETTINGS.max_upload_mb,
    }.items():
        st.session_state[key] = value
    st.session_state.pop("installed_ollama_model", None)
    save_settings(DEFAULT_SETTINGS, _SETTINGS_PATH)


def _navigate(page: str) -> None:
    # The app's active page is independent of the radio's widget state. Queue
    # the radio update so it is applied before that widget is instantiated.
    if page in PAGES:
        st.session_state["_pending_navigation"] = page


def _overview_question_submitted() -> None:
    question = str(st.session_state.get("overview_question", "")).strip()
    if question:
        st.session_state["ask_question"] = question
        _navigate("Ask AI")


def _sidebar_navigation_changed() -> None:
    selected = st.session_state.get("workspace_page_widget")
    if selected in PAGES:
        st.session_state["workspace_page"] = selected


def _navigate_to_review(record_key: str) -> None:
    st.session_state["review_record_key"] = record_key
    _navigate("Citation Review")


def _config_panel() -> DashboardConfig:
    """Render the compact navigation rail and advanced local settings."""
    with st.sidebar:
        st.markdown(
            '<div class="brand-lockup"><div class="brand-mark">LR</div>'
            '<div><div class="brand-name">LocalRAG</div>'
            '<div class="brand-subtitle">Research workspace</div></div></div>',
            unsafe_allow_html=True,
        )
        st.radio("Workspace", PAGES, key="workspace_page_widget", label_visibility="collapsed",
                 on_change=_sidebar_navigation_changed)
        st.divider()

        with st.expander("Advanced settings", expanded=False):
            st.selectbox("Research preset", ("FAST SEARCH", "BALANCED (DEFAULT)", "DEEP RESEARCH", "CUSTOM"),
                         key="preset", on_change=_preset_changed,
                         help="Presets change query depth and timeout only; embedding and chunk settings stay fixed.")
            st.slider("Default retrieved passages", 1, 20, key="query_top_k", on_change=_save_settings_callback)
            enabled = st.checkbox("Use a minimum cosine similarity", key="minimum_enabled",
                                  on_change=_save_settings_callback)
            st.slider("Minimum cosine similarity", min_value=-1.0, max_value=1.0, step=.01, key="minimum_score",
                      disabled=not enabled, on_change=_save_settings_callback)
            st.checkbox("Allow optional Crossref lookup", key="crossref", on_change=_save_settings_callback)
            st.slider("Candidate chunk pool", min_value=int(st.session_state.query_top_k),
                      max_value=200, step=5, key="candidate_chunks",
                      on_change=_save_settings_callback)
            st.divider()
            st.caption("Index-time settings · apply only after confirming a rebuild")
            st.text_input("Embedding model", key="config_embedding_model", on_change=_save_settings_callback)
            st.number_input("Chunk size (words)", min_value=1, max_value=5000, key="config_chunk_size",
                            on_change=_save_settings_callback)
            st.number_input("Chunk overlap (words)", min_value=0,
                            max_value=max(0, int(st.session_state.config_chunk_size) - 1),
                            key="config_chunk_overlap",
                            on_change=_save_settings_callback)
            st.divider()
            documents = Path(st.text_input("Documents folder", key="config_documents", on_change=_save_settings_callback))
            index = Path(st.text_input("FAISS index folder", key="config_index", on_change=_save_settings_callback))
            metadata = Path(st.text_input("Citation review catalog", key="config_metadata", on_change=_save_settings_callback))
            requested_ollama_url = st.session_state.config_ollama_url
            parsed_ollama_url = urlparse(requested_ollama_url)
            local_ollama_url = (parsed_ollama_url.scheme in {"http", "https"}
                                and parsed_ollama_url.hostname in {"localhost", "127.0.0.1", "::1"})
            if not local_ollama_url:
                st.error("Ollama API must use a local loopback URL. LocalRAG will not contact a remote server.")
            status_url = requested_ollama_url if local_ollama_url else DEFAULT_SETTINGS.ollama_url
            status_for_models = _service(DashboardConfig(ollama_url=status_url)).ollama_status(.5)
            local_models = status_for_models["models"]
            if local_models:
                selected_model = st.selectbox("Installed Ollama model", local_models,
                                              index=(local_models.index(st.session_state.config_ollama_model)
                                                     if st.session_state.config_ollama_model in local_models else None),
                                              placeholder="Select an installed model",
                                              on_change=_installed_model_changed, key="installed_ollama_model")
                st.session_state.ollama_model_ready = selected_model is not None
                if selected_model is None:
                    st.caption("Select an installed model before using Ask AI.")
            else:
                st.text_input("Installed Ollama model", value="Ollama offline", disabled=True)
                st.session_state.ollama_model_ready = True
            st.text_input("Ollama API URL", key="config_ollama_url", on_change=_save_settings_callback)
            st.number_input("Ollama timeout (seconds)", min_value=1, max_value=900,
                            key="config_ollama_timeout", on_change=_save_settings_callback)
            st.number_input("Maximum upload (MB)", min_value=1, max_value=500,
                            key="config_upload_limit", on_change=_save_settings_callback)
            if st.button("Reset to defaults", on_click=_reset_settings, type="secondary"):
                st.rerun()
        config = DashboardConfig(documents, index, metadata,
                                 st.session_state.config_embedding_model,
                                 st.session_state.config_ollama_model,
                                 status_url,
                                 float(st.session_state.config_ollama_timeout),
                                 int(st.session_state.config_upload_limit),
                                 int(st.session_state.config_chunk_size),
                                 int(st.session_state.config_chunk_overlap))
        if st.session_state.get("settings_save_error"):
            st.error(st.session_state.settings_save_error)
        if st.session_state.get("settings_save_notice"):
            st.info(st.session_state.settings_save_notice)
        status = status_for_models
        if status["available"]:
            st.markdown('<div class="connection"><span class="dot online"></span>Ollama connected</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="connection"><span class="dot offline"></span>Ollama offline</div>',
                        unsafe_allow_html=True)
        license_link = "[AGPL-3.0-or-later](https://www.gnu.org/licenses/agpl-3.0.html)"
        source_url = os.environ.get("LOCALRAG_SOURCE_URL", "").strip()
        parsed_source = urlparse(source_url)
        st.caption("© 2026 Guilherme Pupini · Free software under AGPL-3.0-or-later · No warranty.")
        if parsed_source.scheme == "https" and parsed_source.hostname:
            st.caption(f"LocalRAG · {license_link} · [Source code]({source_url})")
        else:
            st.caption(f"LocalRAG · {license_link}")
    return config


def _page_header(eyebrow: str, title: str, description: str) -> None:
    st.caption(eyebrow.upper())
    st.title(title)
    st.markdown(f'<div class="page-description">{description}</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str | int, detail: str) -> None:
    with st.container(border=True):
        st.metric(label, value)
        st.caption(detail)


def _overview(service: DashboardService, config: DashboardConfig) -> None:
    stats = service.index_summary()
    _page_header("Your private research desk", "Research, grounded in your library.",
                 "Ask questions across your documents, discover related papers, and keep citations reviewable. Documents, indexing, and generation stay local; Crossref uses a network connection only when you enable it.")
    if not stats["exists"]:
        st.warning(stats["error"])
    elif stats.get("stale"):
        st.warning(stats["error"])

    st.markdown("#### Start with a question")
    st.text_input("Ask your library", placeholder="What do these papers say about CPU scheduling?",
                  label_visibility="collapsed", key="overview_question")
    st.button("Ask LocalRAG  →", type="primary", key="overview-ask",
              on_click=_overview_question_submitted)

    actions = st.columns(3)
    for column, label, page in zip(actions, ("Ask AI", "Find citations", "Upload documents"),
                                   ("Ask AI", "Citation Search", "Documents")):
        with column:
            st.button(f"{label}  →", key=f"quick-{page}", width="stretch",
                      on_click=_navigate, args=(page,))

    st.markdown("#### Library at a glance")
    cards = st.columns(3)
    _metric_card_in_column(cards[0], "Indexed documents", stats["documents"], "Distinct source files")
    _metric_card_in_column(cards[1], "Indexed chunks", stats["chunks"], "Searchable passages")
    _metric_card_in_column(cards[2], "Vector dimensions", stats["dimension"] or "—", "Normalized embeddings")
    if stats["model"]:
        st.caption(f"Embedding model · `{stats['model']}`")
    else:
        st.caption(f"Configured embedding model · `{config.embedding_model}`")

    workspace_files = sorted(service.document_inventory(), key=lambda row: row["modified"], reverse=True)
    st.markdown("#### Documents in your workspace")
    if workspace_files:
        recent = [{"Document": row["name"], "Type": str(row["type"]).upper(),
                   "Size": (f"{row['size_bytes'] / (1024 * 1024):.1f} MB" if row["size_bytes"] >= 1024 * 1024
                            else f"{row['size_bytes'] / 1024:.1f} KB" if row["size_bytes"] >= 1024
                            else f"{row['size_bytes']} B"),
                   "Modified": datetime.fromtimestamp(row["modified"]).strftime("%b %d, %Y"),
                   "Index status": row["status"]}
                  for row in workspace_files[:6]]
        st.dataframe(recent, hide_index=True, width="stretch")
    else:
        st.caption("No local PDF or TXT files yet. Add documents to begin building your research library.")

    with st.expander("Evaluation details"):
        report = service.evaluation_summary()
        if report is None:
            st.info("No evaluation report is available. Run the evaluation workflow to display measured metrics.")
        else:
            label_type = report["label_type"]
            synthetic = "synthetic" in label_type.casefold() or "fixture" in label_type.casefold()
            if synthetic:
                st.info(f"Synthetic fixture benchmark · {report['question_count']} questions · "
                        f"{report['document_count']} fixture documents. These metrics do not evaluate your corpus.")
            else:
                st.caption(f"Dataset type: {label_type} · {report['question_count']} questions")
            if report["configurations"]:
                st.dataframe(report["configurations"], hide_index=True, width="stretch")
            st.caption(f"Report: {report['source']}")


def _metric_card_in_column(column, label: str, value: str | int, detail: str) -> None:
    with column:
        _metric_card(label, value, detail)


def _filename_label(source: str) -> str:
    stem = Path(source).stem.replace("_", " ").replace("-", " ")
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def _persist_query_preferences(top_k: int, minimum: float | None, candidate_chunks: int | None = None,
                              crossref: bool | None = None) -> None:
    # Persist the submitted values without touching keys whose widgets already
    # exist in this run. Their synchronized values are applied at next-run start.
    current = _settings_from_state()
    pool = max(top_k, candidate_chunks if candidate_chunks is not None
               else current.query.candidate_chunks)
    preset = current.query.preset
    query = QuerySettings(preset, top_k, minimum,
                          current.query.crossref if crossref is None else crossref, pool)
    updated = DashboardSettings(
        documents_dir=current.documents_dir, index_dir=current.index_dir,
        metadata_path=current.metadata_path, ollama_model=current.ollama_model,
        ollama_url=current.ollama_url, ollama_timeout=current.ollama_timeout,
        max_upload_mb=current.max_upload_mb, query=query, index=current.index)
    if preset != "CUSTOM" and not matches_named_preset(updated):
        query = QuerySettings("CUSTOM", top_k, minimum, query.crossref, pool)
        updated = DashboardSettings(
            documents_dir=current.documents_dir, index_dir=current.index_dir,
            metadata_path=current.metadata_path, ollama_model=current.ollama_model,
            ollama_url=current.ollama_url, ollama_timeout=current.ollama_timeout,
            max_upload_mb=current.max_upload_mb, query=query, index=current.index)
    save_settings(updated, _SETTINGS_PATH)
    st.session_state["_pending_widget_state"] = {
        "preset": query.preset, "query_top_k": top_k, "ask_top_k": top_k,
        "search_top_k": top_k, "minimum_enabled": minimum is not None,
        "minimum_score": minimum if minimum is not None else .25,
        "ask_filter_minimum": minimum is not None, "search_filter_minimum": minimum is not None,
        "ask_minimum_value": minimum if minimum is not None else .25,
        "search_minimum_value": minimum if minimum is not None else .25,
        "crossref": query.crossref, "search_crossref": query.crossref,
        "candidate_chunks": pool, "search_candidate_chunks": pool,
    }


def _ask_ai(service: DashboardService) -> None:
    _page_header("Grounded answers", "Ask your research library.",
                 "Responses are generated from retrieved passages. Open the evidence cards to inspect every cited source.")
    with st.form("ask_form"):
        question = st.text_area("Research question", placeholder="What do these papers say about…",
                                height=105, key="ask_question")
        with st.expander("Retrieval settings"):
            col1, col2 = st.columns(2)
            top_k = col1.slider("Retrieved passages", 1, 20, key="ask_top_k")
            filter_scores = col2.checkbox("Set a minimum cosine similarity", key="ask_filter_minimum")
            minimum = st.slider("Minimum similarity", min_value=-1.0, max_value=1.0, step=.01,
                                disabled=not filter_scores,
                                key="ask_minimum_value")
        submitted = st.form_submit_button("Ask LocalRAG  →", type="primary")
    if not submitted:
        return
    if not question.strip():
        st.warning("Enter a question first.")
        return
    if not st.session_state.get("ollama_model_ready", True):
        st.error("Select a model reported by the local Ollama server in Advanced settings first. LocalRAG will not pull models automatically.")
        return
    try:
        _persist_query_preferences(top_k, minimum if filter_scores else None)
    except (ValueError, OSError) as error:
        st.error(f"Could not save query settings: {error}")
        return
    try:
        with st.spinner("Retrieving evidence and generating a local answer…"):
            response = service.answer(question, top_k, minimum if filter_scores else None)
    except FileNotFoundError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"Local answer generation failed: {error}")
        st.caption("Check that Ollama is running and the selected model is installed locally.")
        return

    if response.status == "insufficient_evidence":
        st.warning(response.answer)
    elif response.status != "ok":
        st.warning(response.answer)
    else:
        with st.container(border=True):
            st.caption("LOCALRAG ANSWER")
            st.markdown("### Answer")
            st.write(response.answer)
            if response.citations:
                st.caption("Validated source labels · " + " · ".join(f"[{item}]" for item in response.citations))
    if response.scope_note:
        st.info(response.scope_note)
    if response.evidence:
        st.markdown("#### Sources and supporting passages")
        for item in response.evidence:
            page = f" · PDF page {item.page}" if item.page is not None else ""
            with st.expander(f"[{item.source_id}]  {Path(item.source).name}{page}  ·  cosine {item.score:.3f}"):
                st.caption(f"Source: {item.source} · Chunk: {item.chunk_id}")
                st.write(item.text)
    with st.expander("Query timing"):
        cols = st.columns(4)
        for column, key, label in zip(cols, ("embedding", "retrieval", "generation", "total"),
                                      ("Embedding", "FAISS retrieval", "Generation", "Total")):
            column.metric(label, f"{response.timings.get(key, 0):.3f} s")


def _citation_search(service: DashboardService) -> None:
    _page_header("Discover sources", "Search papers by topic.",
                 "Find the best-matching passage in each distinct document. Similarity is a cosine score, not a probability or a measure of citation correctness.")
    with st.form("citation_search_form"):
        query = st.text_input("Research topic", placeholder="e.g. transformer attention mechanisms")
        col1, col2 = st.columns(2)
        top_k = col1.slider("Unique sources", 1, 20, key="search_top_k")
        with st.expander("Search options"):
            candidate_chunks = st.slider("Initial candidate chunk pool",
                                         min_value=int(st.session_state.query_top_k),
                                         max_value=200, step=5, key="search_candidate_chunks")
            use_filter = st.checkbox("Filter by minimum cosine similarity", key="search_filter_minimum")
            minimum = st.slider("Minimum cosine similarity", min_value=-1.0, max_value=1.0, step=.01,
                                disabled=not use_filter,
                                key="search_minimum_value")
            crossref = st.checkbox("Enable optional Crossref lookup", key="search_crossref")
        submitted = st.form_submit_button("Search indexed papers  →", type="primary")
    if not submitted:
        return
    if not query.strip():
        st.warning("Enter a topic to search.")
        return
    try:
        _persist_query_preferences(top_k, minimum if use_filter else None, candidate_chunks, crossref)
    except (ValueError, OSError) as error:
        st.error(f"Could not save search settings: {error}")
        return
    try:
        with st.spinner("Searching the local collection…"):
            results = service.search_sources(query, top_k, candidate_chunks,
                                             minimum if use_filter else None, crossref)
    except Exception as error:
        st.error(f"Citation search failed: {error}")
        return
    if not results:
        st.info("No sources met the current retrieval settings.")
        return
    for item in results:
        with st.container(border=True):
            title_verified = ("title" in item.metadata.verified_fields and
                              item.metadata.provenance.get("title") in {"Crossref", "user-provided"})
            trusted_candidate = bool(item.title and (title_verified or is_plausible_title(item.title)))
            title = item.title if trusted_candidate else _filename_label(item.source)
            head = st.columns([0.72, 0.28])
            with head[0]:
                st.markdown(f"#### {item.rank:02d}  {title}")
                title_state = (("Verified publication title" if title_verified else "Publication title candidate · unverified")
                               if trusted_candidate
                               else "Filename · unverified display label")
                st.caption(title_state)
                st.caption(Path(item.source).name + (f" · PDF page {item.page}" if item.page else ""))
            with head[1]:
                st.metric("Cosine similarity", f"{item.score:.3f}")
                st.caption(f"Bibliographic metadata · {item.status.replace('_', ' ').title()}")
            st.markdown("**Best matching passage**")
            excerpt_limit = 520
            st.write(item.passage[:excerpt_limit] + ("…" if len(item.passage) > excerpt_limit else ""))
            if len(item.passage) > excerpt_limit:
                with st.expander("Expand full retrieved passage"):
                    st.write(item.passage)
            if item.metadata.doi:
                st.caption(f"DOI · {item.metadata.doi}")
            if item.works_cited:
                st.caption("MLA 9 · use the copy control on this citation block")
                st.code(item.works_cited, language=None)
                st.download_button("Download citation", item.works_cited,
                                   file_name=f"citation-{item.rank}.txt", key=f"download-{item.document_id}")
            else:
                missing = list(item.metadata.missing_fields)
                if not item.metadata.kind:
                    missing.append("publication type")
                st.caption("Incomplete or unverified · missing fields: " + (", ".join(missing) or "review required"))
            if item.status in {"ambiguous", "incomplete"}:
                st.button("Review bibliographic metadata", key=f"review-{item.document_id}",
                          on_click=_navigate_to_review, args=(item.document_id,))
            st.caption("Similarity scores rank passages and do not verify citation accuracy.")


def _documents(service: DashboardService, config: DashboardConfig) -> None:
    _page_header("Manage your library", "Your documents, indexed locally.",
                 "Upload papers here, then rebuild the index when you are ready. Uploading does not trigger model work.")
    stats = service.index_summary()
    if not stats["exists"]:
        st.info(stats["error"])
    elif stats.get("stale"):
        st.warning(stats["error"] or "The persisted index is stale.")
    summary = st.columns(3)
    _metric_card_in_column(summary[0], "Indexed documents", stats["documents"], "Distinct source files")
    _metric_card_in_column(summary[1], "Indexed chunks", stats["chunks"], "Searchable passages")
    _metric_card_in_column(summary[2], "Embedding model", stats["model"] or "Not indexed", "Current index model")
    st.caption(f"Local folder: `{config.documents_dir}` · Upload limit: {config.max_upload_mb} MB per file")

    with st.container(border=True):
        st.markdown("#### Add research documents")
        st.caption("PDF and UTF-8 TXT · Files are stored locally and existing filenames are never overwritten.")
        uploads = st.file_uploader("Choose files", type=("pdf", "txt"), accept_multiple_files=True,
                                   max_upload_size=config.max_upload_mb, label_visibility="collapsed")
        if st.button("Save selected files", disabled=not uploads, type="secondary"):
            for upload in uploads or []:
                try:
                    path = service.save_upload(upload.name, upload.getvalue())
                    st.success(f"Saved {path.name}. Rebuild the index to make it searchable.")
                except Exception as error:
                    st.error(f"Could not save {upload.name}: {error}")

    st.markdown("#### Local documents")
    files = service.document_inventory()
    if files:
        def human_size(size: int) -> str:
            return f"{size / (1024 * 1024):.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KB"
        table = [{"Document": item["name"], "Type": str(item["type"]).upper(),
                  "Size": human_size(int(item["size_bytes"])), "Index status": item["status"]}
                 for item in files]
        st.dataframe(table, hide_index=True, width="stretch")
        root = config.documents_dir.resolve(strict=False)
        doc_labels = {str(Path(os.path.abspath(item["path"])).relative_to(root)): item for item in files}
        selected_label = st.selectbox("Preview or manage a document", list(doc_labels))
        selected_doc = doc_labels[selected_label]
        try:
            display_name, file_bytes = service.document_download(str(selected_doc["path"]))
            st.download_button("Download original file", file_bytes, file_name=display_name,
                               mime="application/pdf" if display_name.casefold().endswith(".pdf") else "text/plain",
                               key=f"document-download-{str(selected_doc['path'])}")
            with st.expander("Preview extracted text"):
                st.write(service.document_preview(str(selected_doc["path"])))
        except Exception as error:
            st.error(f"Could not read this document: {error}")
        st.caption("Deleting a document marks the collection stale and blocks search until the index is rebuilt.")
        delete_confirm = st.checkbox(f"I confirm deletion of {selected_doc['name']}",
                                     key=f"delete-confirm-{str(selected_doc['path'])}")
        if st.button("Delete selected document", disabled=not delete_confirm, type="secondary"):
            try:
                deleted = service.delete_document(str(selected_doc["path"]))
                st.success(f"Deleted {deleted.name}. Rebuild the index before searching again.")
                st.rerun()
            except Exception as error:
                st.error(f"Document was not deleted: {error}")
    else:
        st.caption("No PDF or TXT files in the configured folder yet.")
    with st.container(border=True):
        st.markdown("#### Rebuild the search index")
        st.caption("Rebuilding creates and validates a complete replacement generation before it becomes active.")
        confirm_rebuild = st.checkbox("I approve using the configured embedding and chunk settings for this rebuild.",
                                      key="confirm_rebuild_index")
        if st.button("Rebuild Index", type="primary", disabled=not confirm_rebuild):
            progress = st.progress(0.0, text="Preparing ingestion…")
            try:
                service.embedder = _cached_embedder(config.embedding_model)
                result = service.rebuild_index(lambda amount, message: progress.progress(amount, text=message))
                st.success(f"Index rebuilt · {result['documents']} documents · {result['chunks']} chunks · "
                           f"{result['vectors']} vectors · {result['embedding_model']}")
                st.rerun()
            except Exception as error:
                st.error(f"Index rebuild failed: {error}")


def _citation_review(service: DashboardService) -> None:
    _page_header("Bibliography review", "Resolve metadata with a clear audit trail.",
                 "Compare what each source reported, then record an explicit reviewer decision. Approval is not independent source verification.")
    try:
        catalog = service.review_store()
        records = catalog.list_records()
    except Exception as error:
        st.error(f"Could not open citation review catalog: {error}")
        return
    gaps = service.bibliography_gaps()
    if gaps:
        with st.container(border=True):
            st.markdown(f"#### {len(gaps)} indexed source(s) have no bibliography record")
            st.caption("Local text files and PDFs without extracted bibliographic fields are not yet in the review catalog.")
            with st.expander("Sources to scan"):
                for source in gaps:
                    st.write(Path(source).name)
            if st.button("Scan indexed sources and refresh bibliography", type="secondary"):
                try:
                    result = service.refresh_bibliography()
                    st.success(f"Added {result['added']} record(s). Existing approvals and audit history were preserved.")
                    st.rerun()
                except Exception as error:
                    st.error(f"Bibliography scan failed: {error}")
    if not records:
        st.info("No citation metadata has been collected yet. Search papers or scan indexed sources to create review records.")
        return
    counts = {status: sum(record.status == status for record in records)
              for status in ("ambiguous", "incomplete", "manually_reviewed", "verified")}
    columns = st.columns(4)
    for column, status in zip(columns, counts):
        column.metric(status.replace("_", " ").title(), counts[status])
    selected_status = st.selectbox("Filter records", ["All", "ambiguous", "incomplete", "manually_reviewed", "verified"])
    filtered = [record for record in records if selected_status == "All" or record.status == selected_status]
    if not filtered:
        st.info("No records match this status.")
        return
    labels = {f"{Path(row.sources[0]).name if row.sources else row.key} · {row.status}": row
              for row in filtered}
    label_list = list(labels)
    preferred = st.session_state.get("review_record_key")
    preferred_index = next((i for i, label in enumerate(label_list)
                            if labels[label].key == preferred or preferred in labels[label].document_ids), 0)
    if st.session_state.get("bibliography_selected_record") not in label_list:
        st.session_state.bibliography_selected_record = label_list[preferred_index]
    selected = labels[st.selectbox("Bibliographic record", label_list, index=preferred_index,
                                   key="bibliography_selected_record")]
    st.session_state.review_record_key = selected.key
    st.markdown(f"#### {Path(selected.sources[0]).name if selected.sources else selected.key}")
    st.caption(f"Record key: `{selected.key}` · Status: **{selected.status.replace('_', ' ').title()}**")
    st.caption("PDF page references are extracted file page indices. Bibliographic page ranges below refer to the publication only when supplied by verified metadata.")

    effective = selected.effective_metadata()
    missing = list(effective.missing_fields)
    if not effective.kind:
        missing.append("publication type")
    if missing:
        st.warning("MLA 9 is unavailable until these required fields are verified or approved: " + ", ".join(missing))
    fields = sorted(set(selected.candidates) | set(selected.fields) |
                    {"title", "authors", "year", "container", "kind", "volume", "issue", "pages", "doi"})
    if fields:
        st.markdown("##### Field comparison")
        rows = []
        for field in fields:
            candidates = selected.candidates.get(field, ())
            def render_values(provenance: str) -> str:
                values = []
                for candidate in candidates:
                    if candidate.get("provenance") != provenance:
                        continue
                    value = candidate.get("value")
                    value = ", ".join(value) if isinstance(value, list) else str(value)
                    if candidate.get("verification_status") == "rejected_suspicious":
                        value = "[screened: suspicious title] " + value
                    elif field == "title" and provenance == "PDF" and not is_plausible_title(value):
                        value = "[screened: suspicious title] " + value
                    values.append(value[:240] + ("…" if len(value) > 240 else ""))
                return "\n".join(dict.fromkeys(values)) or "—"
            approved = selected.overrides.get(field, ())
            approved_value = approved[-1].get("approved_value") if approved else None
            if isinstance(approved_value, list):
                approved_value = ", ".join(approved_value)
            provenance = ", ".join(dict.fromkeys(str(item.get("provenance", "")) for item in candidates)) or "—"
            status = "Manually reviewed" if approved else selected.status.replace("_", " ").title()
            rows.append({"Field": field.title(), "PDF candidate": render_values("PDF"),
                         "Crossref candidate": render_values("Crossref"),
                         "Approved / current": approved_value or "—",
                         "Provenance": provenance, "Status": status})
        st.dataframe(rows, hide_index=True, width="stretch")
    if selected.original_pdf_metadata:
        with st.expander("Original PDF metadata"):
            st.json(selected.original_pdf_metadata)
    if selected.overrides:
        with st.expander("Approval audit history"):
            st.json(selected.overrides)

    with st.container(border=True):
        st.markdown("#### Approve a correction")
        st.caption("Check the primary source before approving. The saved record preserves prior values and your audit details.")
        field = st.selectbox("Field", ["title", "authors", "year", "journal", "container", "volume", "issue", "pages", "doi", "kind"])
        current_values = selected.candidates.get(field, ())
        eligible = [candidate for candidate in current_values
                    if candidate.get("verification_status") != "rejected_suspicious"
                    and not (field == "title" and candidate.get("provenance") == "PDF"
                             and not is_plausible_title(str(candidate.get("value", ""))))]
        default = json.dumps(eligible[0]["value"], ensure_ascii=False) if len(eligible) == 1 else ""
        if field == "authors":
            correction = st.text_area("Approved complete author list (one name per line)",
                                      value="\n".join(json.loads(default)) if default else "",
                                      key=f"approval-value-{selected.key}-{field}")
            serialized = json.dumps([line.strip() for line in correction.splitlines() if line.strip()], ensure_ascii=False)
        else:
            correction = st.text_input("Approved value", value=(str(json.loads(default)) if default else ""),
                                       key=f"approval-value-{selected.key}-{field}")
            serialized = json.dumps(correction, ensure_ascii=False)
        col1, col2 = st.columns(2)
        source_url = col1.text_input("Verification source URL", placeholder="https://doi.org/…")
        reviewer = col2.text_input("Reviewer")
        confirmation = st.checkbox("I reviewed the source and approve this exact correction.")
        if st.button("Save approved correction", disabled=not confirmation, type="primary"):
            try:
                catalog.approve(selected.document_ids[0] if selected.document_ids else selected.key,
                                field, json.loads(serialized), source_url, reviewer)
                st.success("Correction saved with its original candidates and audit details.")
                st.rerun()
            except Exception as error:
                st.error(f"Correction was not saved: {error}")
    if selected.works_cited:
        st.markdown("##### Regenerated MLA 9")
        st.text_area("Works Cited", value=selected.works_cited, height=82)
        st.download_button("Download MLA entry", selected.works_cited,
                           file_name="works-cited.txt", key=f"review-download-{selected.key}")


def _apply_styles() -> None:
    st.markdown("""
    <style>
      .block-container {max-width: 1180px; padding-top: 2.1rem; padding-bottom: 4rem;}
      [data-testid="stSidebar"] {border-right: 1px solid var(--border-color);}
      .brand-lockup {display:flex;align-items:center;gap:.75rem;padding:.35rem 0 1rem;}
      .brand-mark {display:grid;place-items:center;width:2.55rem;height:2.55rem;border-radius:.8rem;
        color:#fff;background:linear-gradient(135deg,#276b66,#42a99a);font-weight:750;letter-spacing:-.06em;}
      .brand-name {font-size:1.15rem;font-weight:700;line-height:1.2;color:var(--text-color);}
      .brand-subtitle {font-size:.77rem;color:var(--text-color);opacity:.64;margin-top:.15rem;}
      .connection {display:flex;align-items:center;gap:.5rem;padding:.6rem .1rem;color:var(--text-color);font-size:.82rem;}
      .dot {height:.55rem;width:.55rem;border-radius:50%;display:inline-block;}
      .online {background:#2d9b70;box-shadow:0 0 0 3px color-mix(in srgb,#2d9b70 17%,transparent);}
      .offline {background:#ba7842;box-shadow:0 0 0 3px color-mix(in srgb,#ba7842 17%,transparent);}
      h1,h2,h3,h4 {letter-spacing:-.035em;}
      .page-description {max-width:760px;margin-top:-.55rem;margin-bottom:1.6rem;color:var(--text-color);opacity:.72;font-size:1.03rem;line-height:1.65;}
      [data-testid="stMetric"] {background:color-mix(in srgb,var(--secondary-background-color) 68%,transparent);padding:.8rem;border-radius:.65rem;}
      [data-testid="stExpander"] {border-radius:.7rem;}
      [data-testid="stButton"] button {border-radius:.55rem;}
      [data-testid="stTextArea"] textarea,[data-testid="stTextInput"] input {border-radius:.5rem;}
      @media (max-width: 700px) { .block-container {padding:1rem 1rem 3rem;} .page-description {font-size:.96rem;} }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="LocalRAG · Research workspace", page_icon="🔎", layout="wide")
    _apply_styles()
    _initialize_settings()
    st.session_state.setdefault("workspace_page", "Overview")
    st.session_state.setdefault("workspace_page_widget", st.session_state.workspace_page)
    config = _config_panel()
    page = st.session_state.workspace_page
    service = _service(config)
    if page in {"Ask AI", "Citation Search"}:
        stats = service.index_summary()
        if stats["exists"] and stats.get("stale"):
            _page_header("Index update required", "Rebuild before searching.",
                         "The active collection no longer matches the documents or index settings. Rebuild it from Documents; the current persisted generation remains preserved until then.")
            st.warning(stats["error"])
            return
        if stats["exists"]:
            try:
                service.embedder = _cached_embedder(config.embedding_model)
            except Exception as error:
                st.error(f"Embedding model is not available in the local cache: {error}")
                st.caption("Cache the selected Sentence Transformer model before using the dashboard. The app will not download model weights automatically.")
                return
    if page == "Overview":
        _overview(service, config)
    elif page == "Ask AI":
        _ask_ai(service)
    elif page == "Citation Search":
        _citation_search(service)
    elif page == "Documents":
        _documents(service, config)
    else:
        _citation_review(service)


if __name__ == "__main__":
    main()
