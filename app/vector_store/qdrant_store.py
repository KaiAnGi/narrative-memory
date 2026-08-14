"""Almacen vectorial Qdrant.

Soporta dos modos con el mismo interfaz:
- ``local``:  Qdrant embebido en el proceso (implementacion local del propio
  qdrant-client, sin servidor ni binario adicional).
- ``remote``: servidor Qdrant independiente en ``QDRANT_URL``.

En modo local los indices de payload no tienen efecto (el filtrado sigue
funcionando, solo se recorre); se crean unicamente en modo remoto.

Cada punto guarda el payload completo del Chunk (trazabilidad, ajuste #4)
y su vector. La coleccion usa distancia COSINE y tiene indices de payload
para filtrar por book_id, capitulo, chunk y posicion global.
"""
import uuid
from typing import Iterable

from qdrant_client import QdrantClient, models as qm

from app.models.schemas import Chunk, SearchHit

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

INDEXED_PAYLOAD_FIELDS: list[tuple[str, qm.PayloadSchemaType]] = [
    ("book_id", qm.PayloadSchemaType.KEYWORD),
    ("chapter_index", qm.PayloadSchemaType.INTEGER),
    ("chunk_index", qm.PayloadSchemaType.INTEGER),
    ("global_position", qm.PayloadSchemaType.INTEGER),
]


def point_id(book_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{book_id}:{chunk_index}"))


class QdrantStore:
    def __init__(
        self,
        mode: str = "local",
        url: str | None = None,
        path: str | None = None,
        collection_name: str = "narrative_chunks",
    ) -> None:
        if mode == "remote":
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=path)
        self._collection = collection_name
        self._local = mode != "remote"

    def ensure_collection(self, dim: int) -> None:
        if self._client.collection_exists(self._collection):
            size = self._current_dim()
            if size is not None and size != dim:
                raise RuntimeError(
                    f"La coleccion '{self._collection}' existe con dimension {size} "
                    f"y los embeddings dan {dim}. Borrala o usa otra COLLECTION_NAME."
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field, schema in INDEXED_PAYLOAD_FIELDS:
            if self._local:
                continue  # los indices de payload solo tienen efecto en modo remoto
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:  # noqa: BLE001  (indice ya existente)
                pass

    def _current_dim(self) -> int | None:
        try:
            vectors = self._client.get_collection(self._collection).config.params.vectors
        except Exception:  # noqa: BLE001
            return None
        return getattr(vectors, "size", None)

    def delete_book(self, book_id: str) -> None:
        """Borra todos los chunks de un libro (re-ingestion idempotente)."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="book_id", match=qm.MatchValue(value=book_id)
                        )
                    ]
                )
            ),
        )

    def upsert_chunks(
        self, chunks: Iterable[Chunk], vectors: Iterable[list[float]]
    ) -> None:
        points = [
            qm.PointStruct(
                id=point_id(chunk.book_id, chunk.chunk_index),
                vector=vector,
                payload=chunk.payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def search(
        self,
        vector: list[float],
        top_k: int,
        filters: dict | None = None,
        with_vectors: bool = False,
    ) -> list[SearchHit]:
        query_filter = self._build_filter(filters or {})
        resp = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            SearchHit(
                chunk=Chunk(**point.payload),
                score=point.score,
                vector=point.vector if with_vectors else None,
            )
            for point in resp.points
        ]

    def get_chunks(
        self, book_id: str, chunk_indices: Iterable[int]
    ) -> dict[int, Chunk]:
        """Resuelve chunks originales por su chunk_index global (traza de la memoria).

        La memoria narrativa solo localiza: sus ``source_chunks``/``chunk_refs``
        son indices globales; el texto real se obtiene aqui de Qdrant.
        """
        indices = list(chunk_indices)
        if not indices:
            return {}
        resp = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id(book_id, i) for i in indices],
            with_vectors=False,
        )
        return {point.payload["chunk_index"]: Chunk(**point.payload) for point in resp}

    def count(self, book_id: str | None = None) -> int:
        count_filter = None
        if book_id is not None:
            count_filter = qm.Filter(
                must=[
                    qm.FieldCondition(key="book_id", match=qm.MatchValue(value=book_id))
                ]
            )
        return self._client.count(
            collection_name=self._collection, count_filter=count_filter
        ).count

    @staticmethod
    def _build_filter(filters: dict) -> qm.Filter | None:
        must = []
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                must.append(
                    qm.FieldCondition(key=key, match=qm.MatchAny(any=list(value)))
                )
            elif isinstance(value, int):
                must.append(
                    qm.FieldCondition(key=key, match=qm.MatchValue(value=value))
                )
            else:
                must.append(
                    qm.FieldCondition(key=key, match=qm.MatchValue(value=value))
                )
        return qm.Filter(must=must) if must else None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
