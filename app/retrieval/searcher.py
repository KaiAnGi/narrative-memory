"""Recuperacion semantica multi-query con reranking opcional (Fase 1.5).

Flujo:
1. El expansor genera varias consultas a partir de la pregunta (configurable).
2. Cada consulta se embebe y se buscan sus candidatos (por punto en Qdrant).
3. Los candidatos se fusionan por chunk conservando el mejor score.
4. Se seleccionan los top-k por score o por MMR (diversidad de chunks/capitulos).

Sin expansion y sin rerank se reproduce exactamente el comportamiento V1.
"""
from app.models.schemas import SearchResult
from app.retrieval.options import RetrievalOptions
from app.retrieval.query_expander import build_expander
from app.retrieval.reranker import MMRReranker
from app.vector_store.qdrant_store import QdrantStore


class Searcher:
    def __init__(self, store: QdrantStore, embedder, llm=None) -> None:
        self._store = store
        self._embedder = embedder
        self._llm = llm

    def search(
        self,
        query: str,
        top_k: int = 8,
        book_id: str | None = None,
        chapter: int | None = None,
        options: RetrievalOptions | None = None,
    ) -> SearchResult:
        options = options or RetrievalOptions()
        expander = build_expander(
            options.expansion,
            max_queries=options.max_queries,
            llm=self._llm,
        )
        queries = expander.expand(query)

        filters: dict = {}
        if book_id:
            filters["book_id"] = book_id
        if chapter is not None:
            filters["chapter_index"] = chapter

        pool: dict[tuple[str, int], object] = {}
        for q in queries:
            vector = self._embedder.embed([q])[0]
            hits = self._store.search(
                vector,
                options.candidates_per_query,
                filters=filters or None,
                with_vectors=options.rerank == "mmr",
            )
            for hit in hits:
                key = (hit.chunk.book_id, hit.chunk.chunk_index)
                current = pool.get(key)
                if current is None or hit.score > current.score:
                    pool[key] = hit

        merged = sorted(pool.values(), key=lambda h: h.score, reverse=True)

        if options.rerank == "mmr" and len(merged) > 1:
            reranker = MMRReranker(
                diversity_lambda=options.diversity_lambda,
                chapter_penalty=options.chapter_penalty,
            )
            final = reranker.rerank(merged, top_k)
        else:
            final = merged[:top_k]

        return SearchResult(query=query, queries=queries, hits=final)
