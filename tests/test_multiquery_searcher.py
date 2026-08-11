"""Tests del searcher multi-query: fusion de pools y rerank con diversidad.

Reproduce el caso real "aglutinamiento": una consulta unica devuelve dos chunks
casi identicos del mismo capitulo y deja fuera la evidencia de otro capitulo.
"""
import pytest

from app.models.schemas import Chunk
from app.retrieval.options import RetrievalOptions
from app.retrieval.searcher import Searcher
from app.vector_store.qdrant_store import QdrantStore

QUESTION = (
    "¿Cómo evoluciona la relación desde el regreso del museo "
    "hasta las consecuencias del fallo?"
)


def _norm(v: list[float]) -> list[float]:
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def _chunk(book_id, chapter, chunk_index, text):
    return Chunk(
        book_id=book_id,
        chapter_index=chapter,
        chapter_title=f"Capítulo {chapter}",
        chunk_index=chunk_index,
        paragraph_start=chunk_index,
        paragraph_end=chunk_index,
        paragraph_indices=[chunk_index],
        global_position=chunk_index,
        text=text,
    )


class _QueryEmbedder:
    """Simula un embedder: 'regreso' recupera el cap.1, 'consecuencias' el cap.2."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            lower = text.lower()
            if "consecuencias" in lower and "regreso" not in lower:
                vector = _norm([0.15, 0.985, 0])  # favorece el cap.2, score cercano
            elif "regreso" in lower:
                vector = _norm([1, 0.15, 0])      # favorece claramente el cap.1
            else:
                vector = _norm([1, 0.5, 0])
            out.append(vector)
        return out


@pytest.fixture
def store(tmp_path):
    s = QdrantStore(
        mode="local", path=str(tmp_path / "qdrant"), collection_name="test_mq"
    )
    s.ensure_collection(3)
    return s


def _seed(store) -> None:
    chunks = [
        _chunk("b", 1, 0, "el regreso del museo con el narrador"),
        _chunk("b", 1, 1, "siguen de regreso por el mismo camino casi igual"),
        _chunk("b", 2, 2, "las consecuencias del fallo en la habitación"),
        _chunk("b", 2, 3, "más consecuencias del simulador"),
    ]
    vectors = [_norm([1, 0, 0]), _norm([0.99, 0.1, 0]), _norm([0, 1, 0]), _norm([0.5, 0.86, 0])]
    store.upsert_chunks(chunks, vectors)


def _searcher(store) -> Searcher:
    return Searcher(store, _QueryEmbedder())


def test_baseline_single_query_misses_chapter(store):
    _seed(store)
    result = _searcher(store).search(
        QUESTION,
        top_k=2,
        book_id="b",
        options=RetrievalOptions(expansion="off", rerank="none"),
    )
    assert {h.chunk.chapter_index for h in result.hits} == {1}


def test_multiquery_merges_but_score_keeps_crowding(store):
    _seed(store)
    result = _searcher(store).search(
        QUESTION,
        top_k=2,
        book_id="b",
        options=RetrievalOptions(expansion="heuristic", rerank="none"),
    )
    assert len(result.queries) >= 2
    assert result.queries[0] == QUESTION
    assert {h.chunk.chapter_index for h in result.hits} == {1}


def test_multiquery_with_mmr_recovers_both_chapters(store):
    _seed(store)
    result = _searcher(store).search(
        QUESTION,
        top_k=2,
        book_id="b",
        options=RetrievalOptions(expansion="heuristic", rerank="mmr"),
    )
    assert {h.chunk.chapter_index for h in result.hits} == {1, 2}


def test_candidates_per_query_limits_pool(store):
    _seed(store)
    result = _searcher(store).search(
        QUESTION,
        top_k=2,
        book_id="b",
        options=RetrievalOptions(expansion="heuristic", rerank="none", candidates_per_query=1),
    )
    assert len(result.hits) <= 2
