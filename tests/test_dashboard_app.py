# SPDX-License-Identifier: AGPL-3.0-or-later
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def _use_empty_test_workspace(tmp_path, monkeypatch, query=None):
    settings_file = tmp_path / "settings.json"
    config = {"documents_dir": str(tmp_path / "documents"), "index_dir": str(tmp_path / "index")}
    if query is not None:
        config["query"] = query
    settings_file.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("LOCALRAG_DASHBOARD_SETTINGS", str(settings_file))
    return settings_file


def test_streamlit_dashboard_starts_without_an_index():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    assert not app.exception
    assert any("Research, grounded in your library." in item.value for item in app.title)
    assert app.sidebar.radio[0].value == "Overview"
    assert any(item.label == "Advanced settings" for item in app.expander)


def test_dashboard_navigation_renders_documents_and_citation_review():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    app.sidebar.radio[0].set_value("Documents").run()
    assert not app.exception
    assert any("Your documents, indexed locally." in item.value for item in app.title)
    assert any(button.label == "Rebuild Index" for button in app.button)
    assert app.file_uploader[0].proto.max_upload_size_mb == 50

    app.sidebar.radio[0].set_value("Citation Review").run()
    assert not app.exception
    assert any("Resolve metadata with a clear audit trail." in item.value for item in app.title)


def test_ask_and_citation_search_navigation_does_not_load_models_without_index():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    app.sidebar.radio[0].set_value("Ask AI").run()
    assert not app.exception
    assert any("Ask your research library." in item.value for item in app.title)
    app.sidebar.radio[0].set_value("Citation Search").run()
    assert not app.exception
    assert any("Search papers by topic." in item.value for item in app.title)


def test_rebuild_requires_explicit_confirmation():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    app.sidebar.radio[0].set_value("Documents").run()
    button = next(item for item in app.button if item.label == "Rebuild Index")
    assert button.disabled
    confirmation = next(item for item in app.checkbox if "approve using the configured" in item.label)
    confirmation.check().run()
    button = next(item for item in app.button if item.label == "Rebuild Index")
    assert not button.disabled


def test_configured_upload_limit_matches_streamlit_uploader(tmp_path, monkeypatch):
    _use_empty_test_workspace(tmp_path, monkeypatch)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    upload_limit = next(item for item in app.number_input if item.label == "Maximum upload (MB)")
    upload_limit.set_value(5).run()
    app.sidebar.radio[0].set_value("Documents").run()
    assert app.file_uploader[0].proto.max_upload_size_mb == 5


def test_advanced_settings_are_available_without_cluttering_navigation():
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    assert not app.exception
    assert any(item.label == "Advanced settings" for item in app.expander)
    assert list(app.sidebar.radio[0].options) == [
        "Overview", "Ask AI", "Citation Search", "Documents", "Citation Review"
    ]


def test_preset_changes_and_custom_values_persist_across_reruns(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    _use_empty_test_workspace(tmp_path, monkeypatch)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    preset = next(item for item in app.selectbox if item.label == "Research preset")
    preset.select("FAST SEARCH").run()
    assert app.session_state["query_top_k"] == 3
    assert app.session_state["config_ollama_timeout"] == 60
    next(item for item in app.number_input if item.label == "Maximum upload (MB)").set_value(60).run()
    assert app.session_state["preset"] == "FAST SEARCH"
    top_k = next(item for item in app.slider if item.label == "Default retrieved passages")
    top_k.set_value(7).run()
    assert app.session_state["preset"] == "CUSTOM"
    assert settings_file.is_file()

    restarted = AppTest.from_file(str(app_path), default_timeout=10).run()
    assert restarted.session_state["query_top_k"] == 7
    assert restarted.session_state["preset"] == "CUSTOM"


def test_balanced_preset_restores_after_application_restart(tmp_path, monkeypatch):
    from dataclasses import replace
    from localrag.dashboard_settings import DEFAULT_SETTINGS, save_settings, with_preset

    settings_file = tmp_path / "settings.json"
    monkeypatch.setenv("LOCALRAG_DASHBOARD_SETTINGS", str(settings_file))
    settings = replace(with_preset(DEFAULT_SETTINGS, "BALANCED (DEFAULT)"),
                       documents_dir=str(tmp_path / "documents"),
                       index_dir=str(tmp_path / "index"))
    save_settings(settings, settings_file)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    restarted = AppTest.from_file(str(app_path), default_timeout=10).run()
    assert not restarted.exception
    assert next(item for item in restarted.selectbox
                if item.label == "Research preset").value == "BALANCED (DEFAULT)"
    assert restarted.session_state["query_top_k"] == 5


def test_overview_question_action_navigates_before_navigation_widget_renders(tmp_path, monkeypatch):
    _use_empty_test_workspace(tmp_path, monkeypatch)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    next(item for item in app.text_input if item.label == "Ask your library").set_value("Explain indexing").run()
    next(item for item in app.button if item.label.startswith("Ask LocalRAG")).click().run()
    assert not app.exception
    assert app.session_state.get("workspace_page") == "Ask AI", dict(app.session_state)
    assert app.sidebar.radio[0].value == "Ask AI"
    assert app.session_state["ask_question"] == "Explain indexing"


def test_ask_submission_persists_defaults_and_custom_values_without_widget_mutation(tmp_path, monkeypatch):
    from localrag.dashboard_settings import load_settings

    settings_file = _use_empty_test_workspace(tmp_path, monkeypatch, {"minimum_score": 0.25})
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    app.sidebar.radio[0].set_value("Ask AI").run()
    app.text_area[0].set_value("Explain indexing")
    next(item for item in app.form_submit_button if item.label.startswith("Ask LocalRAG")).click().run()
    assert not app.exception
    saved = load_settings(settings_file)
    assert saved.query.top_k == 5
    assert saved.query.preset == "BALANCED (DEFAULT)"
    assert app.session_state["query_top_k"] == 5

    # A fresh browser session restores a customized retrieval configuration.
    _use_empty_test_workspace(tmp_path, monkeypatch,
                              {"preset": "CUSTOM", "top_k": 7, "minimum_score": .4,
                               "candidate_chunks": 30})
    customized = AppTest.from_file(str(app_path), default_timeout=10).run()
    customized.sidebar.radio[0].set_value("Ask AI").run()
    next(item for item in customized.form_submit_button
         if item.label.startswith("Ask LocalRAG")).click().run()
    assert not customized.exception
    assert load_settings(settings_file).query.top_k == 7
    assert customized.session_state["ask_top_k"] == 7


def test_citation_search_submission_persists_filters_and_navigates_back(tmp_path, monkeypatch):
    from localrag.dashboard_settings import load_settings

    settings_file = _use_empty_test_workspace(
        tmp_path, monkeypatch,
        {"minimum_score": 0.25},
    )
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    app.sidebar.radio[0].set_value("Citation Search").run()
    next(item for item in app.text_input if item.label == "Research topic").set_value("CPU management")
    app.slider[0].set_value(4)
    app.slider[1].set_value(40)
    next(item for item in app.slider if item.key == "search_minimum_value").set_value(.3)
    next(item for item in app.checkbox if item.label == "Enable optional Crossref lookup").check()
    next(item for item in app.form_submit_button if item.label.startswith("Search indexed papers")).click().run()
    assert not app.exception
    saved = load_settings(settings_file)
    assert saved.query.top_k == 4
    assert saved.query.minimum_score == .3
    assert saved.query.crossref is True
    assert saved.query.candidate_chunks == 40
    app.sidebar.radio[0].set_value("Overview").run()
    app.sidebar.radio[0].set_value("Citation Search").run()
    assert not app.exception
    assert app.session_state["search_top_k"] == 4


def test_quick_navigation_and_reset_update_widgets_on_next_run(tmp_path, monkeypatch):
    from localrag.dashboard_settings import DEFAULT_SETTINGS, load_settings

    settings_file = tmp_path / "settings.json"
    _use_empty_test_workspace(tmp_path, monkeypatch)
    app_path = Path(__file__).resolve().parents[1] / "streamlit_app.py"
    app = AppTest.from_file(str(app_path), default_timeout=10).run()
    next(item for item in app.button if item.label.startswith("Upload documents")).click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Documents"
    app.sidebar.radio[0].set_value("Overview").run()
    next(item for item in app.button if item.label.startswith("Ask AI")).click().run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Ask AI"

    next(item for item in app.selectbox if item.label == "Research preset").select("DEEP RESEARCH").run()
    assert app.session_state["query_top_k"] == 8
    next(item for item in app.button if item.label == "Reset to defaults").click().run()
    assert not app.exception
    assert load_settings(settings_file) == DEFAULT_SETTINGS
    assert app.session_state["query_top_k"] == DEFAULT_SETTINGS.query.top_k
