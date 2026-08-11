"""Ingesta una novela .docx: extrae, trocea, embebe e indexa en Qdrant.

Uso: python scripts/ingest.py <ruta.docx> [--verbose]

Mide tambien: tiempo total de ingesta, tiempo medio por lote de embeddings,
dimension de los vectores y tamano real de la coleccion.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.ollama_embedder import OllamaEmbedder  # noqa: E402
from app.service import Service  # noqa: E402


class TimedEmbedder(OllamaEmbedder):
    """Igual que OllamaEmbedder pero cronometra cada lote (solo diagnostico)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_times: list[float] = []

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        try:
            return super()._embed_batch(batch)
        finally:
            self.batch_times.append(time.perf_counter() - t0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingesta un .docx en Qdrant")
    parser.add_argument("path", help="Ruta al archivo .docx")
    parser.add_argument("--verbose", action="store_true", help="Lista los capítulos detectados")
    args = parser.parse_args()

    settings = get_settings()
    settings.books_dir.mkdir(parents=True, exist_ok=True)
    embedder = TimedEmbedder(
        base_url=settings.ollama_base_url,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout_seconds,
    )
    service = Service(settings, embedder=embedder)
    try:
        t0 = time.perf_counter()
        report = service.ingest_book(args.path)
        ingest_s = time.perf_counter() - t0
        collection_size = service._store.count(report.book_id)
    finally:
        service.close()

    print(f"Libro indexado:            {report.book_id}")
    print(f"Archivo:                   {report.path}")
    print(f"Párrafos extraídos:        {report.paragraphs}")
    print(f"Capítulos detectados:      {report.chapters}")
    print(f"Chunks indexados:          {report.chunks}")
    print(f"Tamaño de la colección:    {collection_size} chunks (book_id={report.book_id})")
    n_batches = len(embedder.batch_times)
    if n_batches:
        mean_batch_s = sum(embedder.batch_times) / n_batches
        print(f"Lotes de embeddings:       {n_batches}  "
              f"media={mean_batch_s:.2f}s  "
              f"min={min(embedder.batch_times):.2f}s  "
              f"max={max(embedder.batch_times):.2f}s")
        try:
            print(f"Dimensión del embedding:   {embedder.dim()}")
        except RuntimeError:
            pass
    print(f"Tiempo total de ingesta:   {ingest_s:.2f}s")
    if args.verbose:
        from app.ingestion.chapters import detect_chapters
        from app.ingestion.extractor import extract_docx

        paragraphs = extract_docx(args.path)
        for ch in detect_chapters(paragraphs):
            print(f"  - Capítulo {ch.chapter_index}: {ch.title}")


if __name__ == "__main__":
    main()
