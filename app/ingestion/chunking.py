"""Chunking orientado a narrativa.

- Los chunks se construyen por capitulo y en orden narrativo.
- Nunca se corta un parrafo a la mitad.
- ``chunk_tokens`` y ``chunk_overlap`` son valores configurables (ajuste #2);
  por ahora se estiman tokens como len(texto)/4 (aprox. para espanol).
- Cada chunk conserva su posicion completa dentro de la novela (ajuste #4).
"""
import math

from app.models.schemas import Chunk, ChapterSpan, Paragraph


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _build_chunk(
    book_id: str,
    chapter: ChapterSpan,
    chunk_index: int,
    paragraphs: list[Paragraph],
) -> Chunk:
    indices = [p.paragraph_index for p in paragraphs]
    return Chunk(
        book_id=book_id,
        chapter_index=chapter.chapter_index,
        chapter_title=chapter.title,
        chunk_index=chunk_index,
        paragraph_start=min(indices),
        paragraph_end=max(indices),
        paragraph_indices=indices,
        global_position=paragraphs[0].global_position,
        characters=[],  # V1: sin extraccion de personajes (ajuste #5)
        text="\n\n".join(p.text for p in paragraphs),
    )


def chunk_paragraphs(
    paragraphs: list[Paragraph],
    chapters: list[ChapterSpan],
    book_id: str,
    chunk_tokens: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Divide los parrafos en chunks por capitulo, con overlap por parrafos."""
    chunks: list[Chunk] = []
    chunk_index = 0

    for chapter in chapters:
        units = paragraphs[chapter.start_paragraph : chapter.end_paragraph + 1]
        if not units:
            continue

        cursor = 0
        n = len(units)
        while cursor < n:
            # Crecimiento del chunk actual (sin superar el presupuesto de tokens).
            total = 0
            end = cursor
            while end < n:
                if total + estimate_tokens(units[end].text) > chunk_tokens and end > cursor:
                    break
                total += estimate_tokens(units[end].text)
                end += 1
            # Un unico parrafo gigante no se parte; se indexa entero.
            if end == cursor:
                end = cursor + 1
                total = estimate_tokens(units[cursor].text)

            chunks.append(_build_chunk(book_id, chapter, chunk_index, units[cursor:end]))
            chunk_index += 1

            # Siguiente cursor con overlap: repite la cola del chunk actual.
            nxt = end
            overlap_tokens = 0
            while nxt > cursor:
                overlap_tokens += estimate_tokens(units[nxt - 1].text)
                if overlap_tokens >= chunk_overlap:
                    break
                nxt -= 1
            cursor = nxt if nxt > cursor else end  # garantiza progreso

    return chunks
