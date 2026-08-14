"""Estrategia hybrid retrieval (experimental).

Combina dos fuentes que LOCALIZAN chunks originales del libro:

1. **baseline**: busqueda semantica en Qdrant (mismo comportamiento V1,
   ``expansion=off``, ``rerank=none``).
2. **memoria narrativa**: scoring por capitulo sobre las unidades de la memoria
   (summary/events/relationships/characters/locations). Cada capitulo se expande
   a sus chunks originales via ``source_chunks``/``chunk_refs``.

La memoria JAMAS aporta texto: solo indices. El texto final se resuelve siempre
en Qdrant (``QdrantStore.get_chunks``), de modo que la evidencia entregada al
LLM es texto original del libro. No usa planner, multi-query, MMR ni ningun
LLM durante el retrieval.

Fusion (configurable, ``fusion``):
  - ``rrf``   (defecto): Reciprocal Rank Fusion, score = sum 1/(k+rank) por
    fuente. No requiere calibrar escalas entre Qdrant y la memoria.
  - ``score`` : normalizacion min-max por fuente + media ponderada
    (``weight_baseline``/``weight_narrative``).

En ambos casos los candidatos se deduplican por ``(book_id, chunk_index)`` y se
seleccionan los ``top_k`` finales (por defecto 8, el presupuesto del LLM).

Esta detras de la configuracion experimental ``Settings.retrieval_hybrid`` y NO
modifica el comportamiento por defecto del pipeline.
"""
from dataclasses import dataclass, field
from typing import Iterable

from app.models.schemas import Chunk, SearchHit, SearchResult
from app.memory.retrieval import score_question
from app.retrieval.options import RetrievalOptions
from app.retrieval.searcher import Searcher
from app.vector_store.qdrant_store import QdrantStore


@dataclass(frozen=True)
class HybridResult:
    query: str
    result: SearchResult
    contributions: list[dict] = field(default_factory=list)
    narrative_ranked: list[dict] = field(default_factory=list)


def _normalize(scores: Iterable[float]) -> dict[float, float]:
    """Min-max normaliza a [0, 1]. Si el rango es cero, todo vale 1.0."""
    vals = list(scores)
    if not vals:
        return {}
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        return {v: 1.0 for v in vals}
    return {v: (v - lo) / span for v in vals}


def expand_narrative_chapters(
    ranked: list[dict],
    memory_chapters: list[dict],
    top_n: int,
    chunks_per_chapter: int,
) -> list[dict]:
    """Convierte capitulos rankeados por la memoria en chunks originales.

    Cada capitulo aporta hasta ``chunks_per_chapter`` indices, tomados de los
    ``source_chunks`` de la evidencia ganadora (los mas precisos) o, si no
    tiene, de ``chunk_refs`` del capitulo. Devuelve, en orden de capitulo,
    los indices de chunk candidatos.
    """
    by_cap = {c["chapter_index"]: c for c in memory_chapters}
    candidates: list[dict] = []
    for entry in ranked[:top_n]:
        cap = entry["chapter_index"]
        mem = by_cap.get(cap, {})
        evidence = entry.get("evidence") or {}
        source = evidence.get("source_chunks") or mem.get("chunk_refs") or []
        picked = list(source[:chunks_per_chapter])
        for chunk_index in picked:
            candidates.append({
                "chapter_index": cap,
                "chunk_index": chunk_index,
                "narrative_score": entry["score"],
                "evidence": evidence,
            })
    return candidates


class HybridSearcher:
    def __init__(
        self,
        store: QdrantStore,
        embedder,
        units: list[dict],
        baseline_options: RetrievalOptions | None = None,
        narrative_top: int = 8,
        chunks_per_chapter: int = 2,
        fusion: str = "rrf",
        weight_baseline: float = 0.5,
        weight_narrative: float = 0.5,
        rrf_k: float = 60.0,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._units = units
        self._baseline_options = baseline_options or RetrievalOptions(
            expansion="off", rerank="none"
        )
        self._narrative_top = narrative_top
        self._chunks_per_chapter = chunks_per_chapter
        self._fusion = fusion
        self._w_base = weight_baseline
        self._w_narr = weight_narrative
        self._rrf_k = rrf_k
        self._searcher = Searcher(store, embedder)

    def search(
        self,
        query: str,
        top_k: int = 8,
        book_id: str | None = None,
        memory_chapters: list[dict] | None = None,
    ) -> HybridResult:
        # 1. Baseline (Qdrant, comportamiento V1 intacto).
        baseline = self._searcher.search(
            query, top_k=top_k, book_id=book_id, options=self._baseline_options
        )

        # 2. Memoria narrativa: rankeo de capitulos y expansion a chunks.
        q_emb = self._embedder.embed([query])[0]
        ranked = score_question(query, q_emb, self._units)
        narr_candidates = expand_narrative_chapters(
            ranked,
            memory_chapters or [],
            self._narrative_top,
            self._chunks_per_chapter,
        )

        # 3. Resolver texto original de los candidatos narrativos en Qdrant.
        narr_chunks: dict[int, Chunk] = {}
        if narr_candidates:
            narr_chunks = self._store.get_chunks(
                book_id, (c["chunk_index"] for c in narr_candidates)
            )

        # 4. Dedupe por chunk y fusion.
        pool: dict[tuple[str, int], dict] = {}
        for rank, hit in enumerate(baseline.hits, start=1):
            key = (hit.chunk.book_id, hit.chunk.chunk_index)
            pool[key] = {
                "chunk": hit.chunk,
                "base_rank": rank,
                "base_score": hit.score,
                "narr_score": None,
                "chapter_index": hit.chunk.chapter_index,
            }
        for cand in narr_candidates:
            chunk = narr_chunks.get(cand["chunk_index"])
            if chunk is None:
                continue
            key = (chunk.book_id, chunk.chunk_index)
            entry = pool.setdefault(
                key,
                {
                    "chunk": chunk,
                    "base_rank": None,
                    "base_score": None,
                    "narr_score": None,
                    "chapter_index": cand["chapter_index"],
                },
            )
            entry["narr_score"] = cand["narrative_score"]

        if self._fusion == "score":
            self._fuse_score(pool)
        else:
            self._fuse_rrf(pool, ranked)

        merged = []
        for (book_id_k, chunk_idx), entry in pool.items():
            merged.append({
                "chunk": entry["chunk"],
                "fused": entry["fused"],
                "from_baseline": entry["base_rank"] is not None,
                "from_narrative": entry["narr_score"] is not None,
                "book_id": book_id_k,
                "chunk_index": chunk_idx,
            })

        merged.sort(key=lambda e: e["fused"], reverse=True)
        final = merged[:top_k]
        hits = [
            SearchHit(chunk=e["chunk"], score=round(e["fused"], 4)) for e in final
        ]
        return HybridResult(
            query=query,
            result=SearchResult(query=query, queries=baseline.queries, hits=hits),
            contributions=[
                {
                    "book_id": e["book_id"],
                    "chunk_index": e["chunk_index"],
                    "chapter_index": e["chunk"].chapter_index,
                    "fused": round(e["fused"], 4),
                    "from_baseline": e["from_baseline"],
                    "from_narrative": e["from_narrative"],
                    "preview": e["chunk"].text[:200],
                }
                for e in final
            ],
            narrative_ranked=[
                {
                    "chapter_index": c["chapter_index"],
                    "score": c["score"],
                    "evidence_kind": (c.get("evidence") or {}).get("kind"),
                }
                for c in ranked[: self._narrative_top]
            ],
        )

    def _fuse_rrf(self, pool: dict, ranked: list[dict]) -> None:
        """Fusion RRF: 1/(k+rank) por fuente, sumada y ponderada."""
        narr_rank = {c["chapter_index"]: i + 1 for i, c in enumerate(ranked)}
        for entry in pool.values():
            fused = 0.0
            if entry["base_rank"] is not None:
                fused += self._w_base / (self._rrf_k + entry["base_rank"])
            if entry["narr_score"] is not None:
                rank = narr_rank.get(entry["chapter_index"])
                if rank is not None:
                    fused += self._w_narr / (self._rrf_k + rank)
            entry["fused"] = round(fused, 4)

    def _fuse_score(self, pool: dict) -> None:
        """Fusion por score: min-max por fuente + media ponderada."""
        base_vals = [e["base_score"] for e in pool.values() if e["base_score"] is not None]
        narr_vals = [e["narr_score"] for e in pool.values() if e["narr_score"] is not None]
        base_norm = _normalize(base_vals)
        narr_norm = _normalize(narr_vals)
        for entry in pool.values():
            b = base_norm.get(entry["base_score"], 0.0) if entry["base_score"] is not None else 0.0
            n = narr_norm.get(entry["narr_score"], 0.0) if entry["narr_score"] is not None else 0.0
            entry["fused"] = round(self._w_base * b + self._w_narr * n, 4)
