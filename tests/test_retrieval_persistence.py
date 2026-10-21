# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import os
from pathlib import Path

import numpy as np
import pytest

from localrag.chunker import ChunkRecord
from localrag.embeddings import EmbeddingBatch
from localrag.retrieval import FaissVectorStore


def make_store(model_name: str = "test-model") -> FaissVectorStore:
    store = FaissVectorStore(dimension=3, model_name=model_name)
    chunks = (
        ChunkRecord("first text", "a.pdf", 1, "first-id", 0),
        ChunkRecord("second text", "b.txt", None, "second-id", 2),
    )
    vectors = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    store.add(EmbeddingBatch(vectors, chunks))
    return store


def get_generation(directory: Path) -> tuple[str, Path, Path]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    generation = directory / "generations" / manifest["generation"]
    return manifest["generation"], generation / "metadata.json", generation / "index.faiss"


def test_save_load_preserves_search_results_and_vector_metadata(tmp_path: Path) -> None:
    original = make_store()
    expected = original.search([1, 0, 0], top_k=2)

    original.save(tmp_path / "index")
    restored = FaissVectorStore.load(tmp_path / "index", expected_model_name="test-model")

    assert len(restored) == 2
    assert restored.dimension == 3
    assert restored.model_name == "test-model"
    actual = restored.search([1, 0, 0], top_k=2)
    assert [result.chunk for result in actual] == [result.chunk for result in expected]
    assert [result.score for result in actual] == pytest.approx(
        [result.score for result in expected]
    )


def test_save_and_load_empty_index(tmp_path: Path) -> None:
    store = FaissVectorStore(dimension=3, model_name="empty-model")
    store.save(tmp_path / "empty")

    restored = FaissVectorStore.load(tmp_path / "empty")

    assert len(restored) == 0
    assert restored.search([1, 0, 0]) == []
    assert restored.model_name == "empty-model"


def test_load_rejects_incompatible_model(tmp_path: Path) -> None:
    make_store("model-one").save(tmp_path / "index")

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        FaissVectorStore.load(tmp_path / "index", expected_model_name="model-two")


def test_load_rejects_index_dimension_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "index"
    make_store().save(root)
    _, metadata_path, _ = get_generation(root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dimension"] = 4
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="dimension"):
        FaissVectorStore.load(root)


def test_load_rejects_corrupted_metadata_json(tmp_path: Path) -> None:
    root = tmp_path / "index"
    make_store().save(root)
    _, metadata_path, _ = get_generation(root)
    metadata_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="Corrupted metadata JSON"):
        FaissVectorStore.load(root)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [("count", 99, "count"), ("schema_version", 2, "schema version")],
)
def test_load_rejects_metadata_count_and_schema_mismatches(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    root = tmp_path / "index"
    make_store().save(root)
    _, metadata_path, _ = get_generation(root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        FaissVectorStore.load(root)


def test_load_rejects_duplicate_chunk_ids(tmp_path: Path) -> None:
    root = tmp_path / "index"
    make_store().save(root)
    _, metadata_path, _ = get_generation(root)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["chunks"][1]["id"] = metadata["chunks"][0]["id"]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        FaissVectorStore.load(root)


def test_load_rejects_missing_index_file(tmp_path: Path) -> None:
    root = tmp_path / "index"
    make_store().save(root)
    _, _, index_path = get_generation(root)
    index_path.unlink()

    with pytest.raises(ValueError, match="Could not read FAISS index"):
        FaissVectorStore.load(root)


def test_interrupted_manifest_switch_keeps_previous_generation_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "index"
    original = make_store("model-one")
    original.save(root)
    previous_generation, _, _ = get_generation(root)

    real_replace = os.replace

    def fail_manifest_switch(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        if Path(destination) == root / "manifest.json":
            raise OSError("simulated interrupted manifest switch")
        real_replace(source, destination)

    monkeypatch.setattr("localrag.retrieval.os.replace", fail_manifest_switch)
    with pytest.raises(OSError, match="simulated interrupted"):
        make_store("model-two").save(root)

    restored = FaissVectorStore.load(root)
    assert restored.model_name == "model-one"
    assert get_generation(root)[0] == previous_generation


def test_invalid_staged_generation_never_replaces_previous_manifest(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "index"
    make_store("model-one").save(root)
    previous_generation, _, _ = get_generation(root)

    def write_invalid_index(index, path):
        Path(path).write_bytes(b"not a faiss index")

    monkeypatch.setattr("localrag.retrieval.faiss.write_index", write_invalid_index)
    with pytest.raises(RuntimeError):
        make_store("model-two").save(root)

    assert get_generation(root)[0] == previous_generation
    assert FaissVectorStore.load(root).model_name == "model-one"


def test_incomplete_generation_is_not_exposed_by_manifest(tmp_path: Path) -> None:
    root = tmp_path / "index"
    make_store().save(root)
    generation_id = "a" * 32
    incomplete = root / "generations" / generation_id
    incomplete.mkdir()
    (root / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "generation": generation_id}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not read index metadata"):
        FaissVectorStore.load(root)
