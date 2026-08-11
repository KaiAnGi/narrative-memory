"""Reranking con diversidad (MMR) para retrieval.

Maximal Marginal Relevance adaptada a narrativa:

- relevancia:  score coseno normalizado del candidato (frente al pool).
- novedad:     1 - maxima similitud con los chunks ya seleccionados (en el
  espacio de embeddings). Evita que 8 chunks casi identicos del mismo capitulo
  expulsen evidencia de otros capitulos.
- capitulos:   los chunks de capitulos aun no representados reciben un bonus de
  novedad; si el capitulo ya tiene chunks seleccionados, la novedad se
  multiplica por ``chapter_penalty`` (diversidad de capitulos).

El resultado final pesa relevancia y diversidad con ``diversity_lambda``.
Generico: no depende del libro ni de capitulos concretos.
"""
from typing import Sequence

from app.models.schemas import SearchHit


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


class MMRReranker:
    def __init__(self, diversity_lambda: float = 0.7, chapter_penalty: float = 0.5) -> None:
        self._lambda = max(0.0, min(1.0, diversity_lambda))
        self._chapter_penalty = chapter_penalty

    def rerank(self, candidates: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        if top_k <= 0 or not candidates:
            return list(candidates)[: max(0, top_k)]

        lo = min(c.score for c in candidates)
        span = (max(c.score for c in candidates) - lo) or 1.0

        pool = list(candidates)
        selected: list[SearchHit] = []
        selected_chapters: set[int] = set()

        while len(selected) < top_k and pool:
            best_index = 0
            best_value = -1.0
            for i, candidate in enumerate(pool):
                relevance = (candidate.score - lo) / span
                sim_selected = 0.0
                if selected:
                    vectors = [s.vector for s in selected if s.vector]
                    if vectors:
                        sim_selected = max(_cosine(candidate.vector, v) for v in vectors)
                novelty = 1.0 - sim_selected
                if candidate.chunk.chapter_index in selected_chapters:
                    novelty *= self._chapter_penalty
                value = self._lambda * relevance + (1.0 - self._lambda) * novelty
                if value > best_value:
                    best_value = value
                    best_index = i
            hit = pool.pop(best_index)
            selected.append(hit)
            selected_chapters.add(hit.chunk.chapter_index)

        return selected
