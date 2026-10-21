# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import pytest

from localrag.chunker import ChunkRecord
from localrag.embeddings import EmbeddingBatch
from localrag.retrieval import FaissVectorStore


def chunk(text: str, chunk_id: str, page: int | None = None) -> ChunkRecord:
    return ChunkRecord(text, "manual.pdf", page, chunk_id, 0)


def batch(vectors: list[list[float]], chunks: list[ChunkRecord]) -> EmbeddingBatch:
    return EmbeddingBatch(np.asarray(vectors, dtype=np.float32), tuple(chunks))


def test_search_returns_nearest_chunk_and_preserves_metadata() -> None:
    store = FaissVectorStore(dimension=3)
    chunks = [chunk("north", "north-id", 4), chunk("east", "east-id", 5)]
    store.add(batch([[2, 0, 0], [0, 3, 0]], chunks))

    results = store.search([1, 0, 0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk == chunks[0]
    assert results[0].chunk.id == "north-id"
    assert results[0].chunk.page == 4
    assert results[0].score == pytest.approx(1.0)


def test_search_orders_by_cosine_similarity_and_caps_to_index_size() -> None:
    store = FaissVectorStore(dimension=2)
    store.add(
        batch(
            [[1, 0], [1, 1], [-1, 0]],
            [chunk("aligned", "a"), chunk("diagonal", "b"), chunk("opposite", "c")],
        )
    )

    results = store.search([1, 0], top_k=10)

    assert [result.chunk.id for result in results] == ["a", "b", "c"]
    assert [result.score for result in results] == pytest.approx([1.0, 2**-0.5, -1.0])


def test_empty_store_returns_no_results() -> None:
    assert FaissVectorStore(dimension=3).search([1, 0, 0]) == []


def test_empty_batch_addition_is_a_noop() -> None:
    store = FaissVectorStore(dimension=3)
    store.add(EmbeddingBatch(np.empty((0, 3), dtype=np.float32), ()))
    assert len(store) == 0


@pytest.mark.parametrize("dimension", [0, -2])
def test_rejects_invalid_index_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension"):
        FaissVectorStore(dimension)


def test_rejects_wrong_embedding_dimension() -> None:
    store = FaissVectorStore(dimension=3)
    with pytest.raises(ValueError, match="shape"):
        store.add(batch([[1, 0]], [chunk("bad", "bad")]))


def test_rejects_zero_embedding() -> None:
    store = FaissVectorStore(dimension=2)
    with pytest.raises(ValueError, match="zero-length"):
        store.add(batch([[0, 0]], [chunk("zero", "zero")]))


@pytest.mark.parametrize("top_k", [-1, 1.5])
def test_rejects_invalid_top_k(top_k: int) -> None:
    store = FaissVectorStore(dimension=2)
    store.add(batch([[1, 0]], [chunk("text", "one")]))
    with pytest.raises(ValueError, match="top_k"):
        store.search([1, 0], top_k=top_k)


class FakeEmbedder:
    dimension = 2

    def embed(self, chunks: list[ChunkRecord]) -> EmbeddingBatch:
        vectors = np.asarray([[1.0, 0.0] for _ in chunks], dtype=np.float32)
        return EmbeddingBatch(vectors, tuple(chunks))


def test_search_text_embeds_query_and_returns_relevant_chunk() -> None:
    store = FaissVectorStore(dimension=2)
    target = chunk("semantic match", "match-id", 8)
    store.add(batch([[1, 0], [0, 1]], [target, chunk("unrelated", "other")]))

    results = store.search_text("find this", FakeEmbedder(), top_k=1)

    assert results[0].chunk == target
    assert results[0].score == pytest.approx(1.0)


def test_search_text_rejects_blank_queries() -> None:
    store = FaissVectorStore(dimension=2)
    with pytest.raises(ValueError, match="query"):
        store.search_text(" \t", FakeEmbedder())
