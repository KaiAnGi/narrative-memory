"""Tests de la estrategia hybrid (Fase 2C).

La memoria narrativa solo localiza (source_chunks/chunk_refs); el texto final se
resuelve en Qdrant. Se verifica el dedupe, la fusion RRF/score y que el hybrid
recupera capitulos que el baseline por si solo pierde.
"""
import pytest

from app.models.schemas import Chunk
from app.retrieval.hybrid import HybridSearcher, expand_narrative_chapters
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
    """El baseline favorece claramente el cap.1 (aglutinamiento)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            if "regreso" in text.lower() and "consecuencias" not in text.lower():
                out.append(_norm([0.9, 0.3, 0]))
            else:
                out.append(_norm([0.9, 0.15, 0]))
        return out


@pytest.fixture
def store(tmp_path):
    s = QdrantStore(
        mode="local", path=str(tmp_path / "qdrant"), collection_name="test_hybrid"
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
    vectors = [
        _norm([0.9, 0.3, 0]),
        _norm([0.8, 0.4, 0]),
        _norm([0.1, 0.9, 0]),
        _norm([0.2, 0.8, 0]),
    ]
    store.upsert_chunks(chunks, vectors)


def _units() -> list[dict]:
    # Las unidades de la memoria se EMBEBEN de forma distinta a los chunks: la
    # memoria debe rankear el cap.2 primero aunque el baseline prefiera el cap.1.
    return [
        {
            "chapter_index": 1,
            "kind": "summary",
            "text": "resumen del regreso del museo",
            "source_chunks": [0, 1],
            "embedding": _norm([0.1, 0.1, 0.985]),
        },
        {
            "chapter_index": 2,
            "kind": "summary",
            "text": "resumen de las consecuencias",
            "source_chunks": [2, 3],
            "embedding": _norm([0.7, 0.7, 0.15]),
        },
    ]


def _memory() -> dict:
    return {
        "book_id": "b",
        "chapters": [
            {
                "chapter_index": 1,
                "summary": "resumen del regreso del museo",
                "chunk_refs": [0, 1],
                "events": [],
                "relationships": [],
                "characters": [],
                "locations": [],
            },
            {
                "chapter_index": 2,
                "summary": "resumen de las consecuencias",
                "chunk_refs": [2, 3],
                "events": [],
                "relationships": [],
                "characters": [],
                "locations": [],
            },
        ],
    }


def _searcher(store, fusion="rrf", chunks_per_chapter=1) -> HybridSearcher:
    return HybridSearcher(
        store,
        _QueryEmbedder(),
        _units(),
        narrative_top=8,
        chunks_per_chapter=chunks_per_chapter,
        fusion=fusion,
    )


def test_get_chunks_resolves_by_chunk_index(store):
    _seed(store)
    got = store.get_chunks("b", [0, 2])
    assert set(got) == {0, 2}
    assert got[0].chapter_index == 1
    assert got[2].chapter_index == 2
    assert "consecuencias" in got[2].text


def test_get_chunks_skips_unknown_index(store):
    _seed(store)
    got = store.get_chunks("b", [0, 999])
    assert set(got) == {0}


def test_baseline_alone_misses_chapter2(store):
    _seed(store)
    from app.retrieval.options import RetrievalOptions
    from app.retrieval.searcher import Searcher

    result = Searcher(store, _QueryEmbedder()).search(
        QUESTION,
        top_k=2,
        book_id="b",
        options=RetrievalOptions(expansion="off", rerank="none"),
    )
    assert {h.chunk.chapter_index for h in result.hits} == {1}


def test_hybrid_rrf_recovers_both_chapters(store):
    _seed(store)
    hr = _searcher(store).search(QUESTION, top_k=2, book_id="b", memory_chapters=_memory()["chapters"])
    assert {h.chunk.chapter_index for h in hr.result.hits} == {1, 2}


def test_hybrid_dedupes_and_budget(store):
    _seed(store)
    hr = _searcher(store).search(QUESTION, top_k=2, book_id="b", memory_chapters=_memory()["chapters"])
    chunk_ids = [c["chunk_index"] for c in hr.contributions]
    assert len(chunk_ids) == 2
    assert len(set(chunk_ids)) == 2


def test_hybrid_contributions_tag_sources(store):
    _seed(store)
    hr = _searcher(store).search(QUESTION, top_k=2, book_id="b", memory_chapters=_memory()["chapters"])
    by_chunk = {c["chunk_index"]: c for c in hr.contributions}
    # chunk 0 esta en ambas fuentes (baseline + memoria); chunk 2 solo en memoria.
    assert by_chunk[0]["from_baseline"] and by_chunk[0]["from_narrative"]
    assert by_chunk[2]["from_narrative"] and not by_chunk[2]["from_baseline"]


def test_hybrid_score_fusion_also_recovers(store):
    _seed(store)
    hr = _searcher(store, fusion="score", chunks_per_chapter=2).search(
        QUESTION, top_k=2, book_id="b", memory_chapters=_memory()["chapters"]
    )
    assert {h.chunk.chapter_index for h in hr.result.hits} == {1, 2}


def test_expand_narrative_chapters_uses_evidence_source_chunks():
    ranked = [
        {
            "chapter_index": 2,
            "score": 0.9,
            "evidence": {"source_chunks": [2, 3], "kind": "summary"},
        },
        {
            "chapter_index": 1,
            "score": 0.5,
            "evidence": {"source_chunks": [0, 1], "kind": "summary"},
        },
    ]
    memory = _memory()["chapters"]
    cands = expand_narrative_chapters(ranked, memory, top_n=8, chunks_per_chapter=1)
    assert [(c["chapter_index"], c["chunk_index"]) for c in cands] == [(2, 2), (1, 0)]
    assert all(c["narrative_score"] in (0.9, 0.5) for c in cands)


def test_expand_narrative_chapters_falls_back_to_chunk_refs():
    ranked = [
        {
            "chapter_index": 1,
            "score": 0.4,
            "evidence": {"kind": "characters", "source_chunks": []},
        }
    ]
    cands = expand_narrative_chapters(ranked, _memory()["chapters"], top_n=8, chunks_per_chapter=2)
    assert [c["chunk_index"] for c in cands] == [0, 1]
