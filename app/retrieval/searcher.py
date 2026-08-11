"""Recuperacion semantica con filtros de metadata.

V1: busqueda por similitud de embeddings + filtro opcional por libro/capitulo.
La infraestructura esta preparada (sin implementar aun) para hybrid search,
reranking y consultas multiples en fases posteriores.
"""
from app.models.schemas import SearchResult
from app.vector_store.qdrant_store import QdrantStore


class Searcher:
    def __init__(self, store: QdrantStore, embedder) -> None:
        self._store = store
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 8,
        book_id: str | None = None,
        chapter: int | None = None,
    ) -> SearchResult:
        vector = self._embedder.embed([query])[0]
        filters: dict = {}
        if book_id:
            filters["book_id"] = book_id
        if chapter is not None:
            filters["chapter_index"] = chapter
        hits = self._store.search(vector, top_k, filters=filters or None)
        return SearchResult(query=query, hits=hits)
