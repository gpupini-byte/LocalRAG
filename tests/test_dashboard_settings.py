# SPDX-License-Identifier: AGPL-3.0-or-later
import pytest

from localrag.dashboard_settings import (DEFAULT_SETTINGS, DashboardSettings, IndexSettings,
                                         QuerySettings, load_settings, matches_named_preset,
                                         save_settings, with_preset)


@pytest.mark.parametrize(
    ("preset", "top_k", "timeout", "candidate_chunks"),
    [("FAST SEARCH", 3, 60, 15), ("BALANCED (DEFAULT)", 5, 120, 25),
     ("DEEP RESEARCH", 8, 180, 60)],
)
def test_presets_change_query_depth_and_timeout_only(preset, top_k, timeout, candidate_chunks):
    custom_index = IndexSettings("local-embedding", 400, 80)
    settings = DashboardSettings(index=custom_index, query=QuerySettings(crossref=True))
    effective = with_preset(settings, preset)
    assert effective.query.top_k == top_k
    assert effective.ollama_timeout == timeout
    assert effective.query.candidate_chunks == candidate_chunks
    assert effective.query.crossref is False
    assert effective.index == custom_index


def test_custom_settings_persist_and_reset_to_default_values(tmp_path):
    path = tmp_path / "dashboard-settings.json"
    custom = DashboardSettings(
        documents_dir="library/papers",
        ollama_timeout=75,
        query=QuerySettings("CUSTOM", 6, .3, True, 40),
        index=IndexSettings("local-embedding", 300, 50),
    ).validate()
    save_settings(custom, path)
    assert load_settings(path) == custom
    assert load_settings(tmp_path / "missing.json") == DEFAULT_SETTINGS


@pytest.mark.parametrize("preset", ["FAST SEARCH", "BALANCED (DEFAULT)", "DEEP RESEARCH"])
def test_named_presets_round_trip_without_becoming_custom(tmp_path, preset):
    path = tmp_path / f"{preset.split()[0].lower()}.json"
    selected = with_preset(DEFAULT_SETTINGS, preset)
    assert matches_named_preset(selected)
    save_settings(selected, path)
    restored = load_settings(path)
    assert restored.query.preset == preset
    assert matches_named_preset(restored)


def test_non_preset_setting_changes_do_not_lose_named_preset_but_query_override_does():
    selected = with_preset(DEFAULT_SETTINGS, "BALANCED (DEFAULT)")
    changed_upload = DashboardSettings(max_upload_mb=10, query=selected.query,
                                       index=selected.index)
    assert matches_named_preset(changed_upload)
    overridden = DashboardSettings(query=QuerySettings("BALANCED (DEFAULT)", 6, None, False, 25),
                                   ollama_timeout=selected.ollama_timeout)
    assert not matches_named_preset(overridden)


def test_invalid_chunk_settings_are_rejected():
    with pytest.raises(ValueError, match="Chunk settings"):
        DashboardSettings(index=IndexSettings(chunk_size=20, overlap=20)).validate()


def test_remote_ollama_api_urls_are_rejected():
    with pytest.raises(ValueError, match="loopback"):
        DashboardSettings(ollama_url="https://example.org/api/chat").validate()
