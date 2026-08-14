"""Orquestacion de la V1: ingestion de un libro y respuesta a preguntas.

Pipeline: DOCX -> extract -> capitulos -> chunking -> embeddings -> Qdrant
        -> búsqueda semántica -> contexto -> Qwen3 -> respuesta.
"""
from pathlib import Path

from app.core.config import Settings
from app.embeddings.ollama_embedder import OllamaEmbedder
from app.ingestion.chapters import detect_chapters
from app.ingestion.chunking import chunk_paragraphs
from app.ingestion.extractor import extract_docx, slugify
from app.llm.ollama_llm import OllamaLLM
from app.llm.prompts import build_qa_messages
from app.models.schemas import Answer, IngestReport, SearchResult
from app.retrieval.hybrid import HybridResult, HybridSearcher
from app.retrieval.options import RetrievalOptions
from app.retrieval.searcher import Searcher
from app.vector_store.qdrant_store import QdrantStore


class Service:
    def __init__(
        self,
        settings: Settings,
        store: QdrantStore | None = None,
        embedder: OllamaEmbedder | None = None,
        llm: OllamaLLM | None = None,
    ) -> None:
        self._settings = settings
        self._store = store or QdrantStore(
            mode=settings.qdrant_mode,
            url=settings.qdrant_url,
            path=str(settings.qdrant_local_path),
            collection_name=settings.collection_name,
        )
        self._embedder = embedder or OllamaEmbedder(
            base_url=settings.ollama_base_url,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            timeout=settings.embedding_timeout_seconds,
        )
        self._llm = llm or OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            think=settings.llm_think,
        )
        self._searcher = Searcher(self._store, self._embedder, llm=self._llm)

    def ingest_book(self, path: str | Path) -> IngestReport:
        """Extrae, trocea, embebe e indexa una novela en Qdrant."""
        settings = self._settings
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo: {path}")

        paragraphs = extract_docx(path)
        if not paragraphs:
            raise ValueError(f"El documento no contiene texto: {path}")

        book_id = slugify(path.stem)
        chapters = detect_chapters(paragraphs)
        chunks = chunk_paragraphs(
            paragraphs,
            chapters,
            book_id,
            chunk_tokens=settings.chunk_tokens,
            chunk_overlap=settings.chunk_overlap,
        )

        # Se indexa en lotes; con el primer lote se crea la coleccion.
        dim: int | None = None
        texts = [c.text for c in chunks]
        for i in range(0, len(texts), settings.embedding_batch_size):
            batch_texts = texts[i : i + settings.embedding_batch_size]
            vectors = self._embedder.embed(batch_texts)
            if dim is None:
                dim = len(vectors[0])
                self._store.ensure_collection(dim)
                self._store.delete_book(book_id)  # re-ingestion idempotente
            self._store.upsert_chunks(chunks[i : i + settings.embedding_batch_size], vectors)

        self._remember_last_book(book_id)
        return IngestReport(
            book_id=book_id,
            path=str(path),
            paragraphs=len(paragraphs),
            chapters=len(chapters),
            chunks=len(chunks),
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        book_id: str | None = None,
        chapter: int | None = None,
        options: RetrievalOptions | None = None,
    ) -> SearchResult:
        top_k = top_k or self._settings.top_k
        options = options or RetrievalOptions.from_settings(self._settings)
        return self._searcher.search(
            query, top_k, book_id=book_id, chapter=chapter, options=options
        )

    def search_hybrid(
        self,
        query: str,
        units: list[dict],
        memory_chapters: list[dict],
        top_k: int | None = None,
        book_id: str | None = None,
        baseline_options: RetrievalOptions | None = None,
    ) -> HybridResult:
        """Retrieval hybrid experimental (Fase 2C): Qdrant + memoria narrativa.

        La memoria solo localiza chunks; el texto final se resuelve en Qdrant.
        No cambia el comportamiento del pipeline: el baseline sigue intacto.
        """
        settings = self._settings
        top_k = top_k or settings.top_k
        searcher = HybridSearcher(
            self._store,
            self._embedder,
            units,
            baseline_options=baseline_options,
            narrative_top=settings.hybrid_narrative_top,
            chunks_per_chapter=settings.hybrid_chunks_per_chapter,
            fusion=settings.hybrid_fusion,
            weight_baseline=settings.hybrid_weight_baseline,
            weight_narrative=settings.hybrid_weight_narrative,
        )
        return searcher.search(
            query, top_k=top_k, book_id=book_id, memory_chapters=memory_chapters
        )

    def ask_question(
        self,
        question: str,
        top_k: int | None = None,
        book_id: str | None = None,
        chapter: int | None = None,
        options: RetrievalOptions | None = None,
    ) -> Answer:
        top_k = top_k or self._settings.top_k
        result = self.search(
            question, top_k, book_id=book_id, chapter=chapter, options=options
        )
        messages = build_qa_messages(question, result.hits)
        answer = self._llm.chat(messages)
        used_chapters = sorted({h.chunk.chapter_index for h in result.hits})
        return Answer(
            question=question,
            answer=answer,
            hits=result.hits,
            used_chapters=used_chapters,
        )

    def _remember_last_book(self, book_id: str) -> None:
        self._settings.books_dir.mkdir(parents=True, exist_ok=True)
        (self._settings.books_dir / ".last_book").write_text(book_id, encoding="utf-8")

    def last_book_id(self) -> str | None:
        marker = self._settings.books_dir / ".last_book"
        if marker.exists():
            return marker.read_text(encoding="utf-8").strip()
        return None

    def close(self) -> None:
        """Cierra el almacen vectorial de forma limpia (evita ruido al salir)."""
        try:
            self._store.close()
        except Exception:  # noqa: BLE001
            pass
