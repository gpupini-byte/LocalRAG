# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from localrag.chunker import ChunkRecord
from localrag.embeddings import SentenceTransformerEmbedder


class FakeSentenceTransformer:
    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.encoded_batches: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
        self.encoded_batches.append(texts)
        rows = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            vector[0] = len(text)
            vector[1] = sum(text.encode("utf-8"))
            rows.append(vector)
        return np.asarray(rows, dtype=np.float32)


def make_chunk(text: str, position: int = 0) -> ChunkRecord:
    return ChunkRecord(text, "source.pdf", 2, f"chunk-{position}", position)


def make_embedder(monkeypatch: pytest.MonkeyPatch, model: FakeSentenceTransformer) -> SentenceTransformerEmbedder:
    import localrag.embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "SentenceTransformer", lambda name, **kwargs: model)
    return SentenceTransformerEmbedder("fake-model")


def test_empty_input_has_expected_empty_shape_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    result = make_embedder(monkeypatch, model).embed([])

    assert result.embeddings.shape == (0, 384)
    assert result.embeddings.dtype == np.float32
    assert result.chunks == ()
    assert model.encoded_batches == []


def test_batch_vectors_are_normalized_and_aligned_with_chunk_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeSentenceTransformer()
    chunks = [make_chunk("first chunk", 0), make_chunk("second chunk", 1)]

    result = make_embedder(monkeypatch, model).embed(chunks)

    assert result.embeddings.shape == (2, 384)
    assert result.embeddings.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(result.embeddings, axis=1), [1.0, 1.0])
    assert result.chunks == tuple(chunks)
    assert [chunk.id for chunk in result.chunks] == ["chunk-0", "chunk-1"]
    assert result.chunks[0].source == "source.pdf"
    assert result.chunks[0].page == 2
    assert model.encoded_batches == [["first chunk", "second chunk"]]


def test_model_instance_is_reused_for_multiple_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    embedder = make_embedder(monkeypatch, model)

    embedder.embed([make_chunk("first")])
    embedder.embed([make_chunk("second")])

    assert len(model.encoded_batches) == 2


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_rejects_blank_chunk_text(monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    embedder = make_embedder(monkeypatch, FakeSentenceTransformer())

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        embedder.embed([make_chunk(text)])


def test_output_dimension_comes_from_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer(dimension=8)
    result = make_embedder(monkeypatch, model).embed([make_chunk("dimensions")])

    assert result.embeddings.shape == (1, 8)


def test_prefers_current_embedding_dimension_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class CurrentAPIFake(FakeSentenceTransformer):
        def get_embedding_dimension(self) -> int:
            return 8

        def get_sentence_embedding_dimension(self) -> int:
            return 99

    embedder = make_embedder(monkeypatch, CurrentAPIFake())

    assert embedder.dimension == 8


def test_rejects_model_output_with_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer(dimension=8)

    def wrong_encode(texts: list[str], **kwargs: object) -> np.ndarray:
        return np.ones((len(texts), 7), dtype=np.float32)

    model.encode = wrong_encode  # type: ignore[method-assign]
    embedder = make_embedder(monkeypatch, model)

    with pytest.raises(ValueError, match=r"expected \(1, 8\)"):
        embedder.embed([make_chunk("dimensions")])
