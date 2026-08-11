"""Tests del reranker MMR (diversidad de chunks y de capitulos)."""
from app.models.schemas import Chunk, SearchHit
from app.retrieval.reranker import MMRReranker


def _hit(book_id, chunk_index, chapter_index, score, vector):
    chunk = Chunk(
        book_id=book_id,
        chapter_index=chapter_index,
        chapter_title=f"Capítulo {chapter_index}",
        chunk_index=chunk_index,
        paragraph_start=chunk_index,
        paragraph_end=chunk_index,
        paragraph_indices=[chunk_index],
        global_position=chunk_index,
        text=f"texto capítulo {chapter_index} chunk {chunk_index}",
    )
    return SearchHit(chunk=chunk, score=score, vector=vector)


def _vec(*components):
    v = list(components)
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def test_mmr_promotes_unrepresented_chapter():
    # 6 chunks del capitulo 1 (muy similares entre si) + 1 chunk del capitulo 2.
    candidates = []
    for i in range(6):
        candidates.append(_hit("b", i, 1, 0.60 - i * 0.01, _vec(1, i * 0.05)))
    candidates.append(_hit("b", 6, 2, 0.575, _vec(0, 1)))  # capitulo 2, score menor
    selected = MMRReranker().rerank(candidates, top_k=3)
    chapters = [h.chunk.chapter_index for h in selected]
    # El capitulo 2 entra en el top-3 aunque tenia menor score.
    assert 2 in chapters
    assert chapters[0] == 1  # el mejor por relevancia sigue primero


def test_mmr_lambda_one_is_pure_relevance():
    candidates = [
        _hit("b", 0, 1, 0.9, _vec(1, 0, 0)),
        _hit("b", 1, 2, 0.8, _vec(0, 1, 0)),
        _hit("b", 2, 1, 0.7, _vec(1, 0.1, 0)),
    ]
    selected = MMRReranker(diversity_lambda=1.0).rerank(candidates, top_k=3)
    assert [h.chunk.chunk_index for h in selected] == [0, 1, 2]


def test_mmr_top_k_limits_output():
    candidates = [_hit("b", i, i + 1, 0.9 - i * 0.1, _vec(1, i)) for i in range(5)]
    selected = MMRReranker().rerank(candidates, top_k=2)
    assert len(selected) == 2


def test_mmr_empty_and_zero():
    assert MMRReranker().rerank([], top_k=3) == []
    assert MMRReranker().rerank([], top_k=0) == []


def test_mmr_chapter_penalty_prefers_new_chapter_on_ties():
    # Dos chunks con el mismo score y misma relevancia normalizada.
    same_vec = _vec(1, 0)
    a = _hit("b", 0, 1, 0.8, same_vec)
    b = _hit("b", 1, 1, 0.8, same_vec)
    c = _hit("b", 2, 2, 0.8, _vec(0, 1))
    selected = MMRReranker().rerank([a, b, c], top_k=2)
    chapters = {h.chunk.chapter_index for h in selected}
    assert chapters == {1, 2}
