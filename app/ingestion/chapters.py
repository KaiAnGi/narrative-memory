"""Deteccion de capitulos.

Estrategia (de mayor a menor fiabilidad):
1. Estilos de encabezado de Word (Heading 1/2, Titulo, Encabezado...).
2. Fallback por regex en el texto de parrafos normales ("Capitulo N"/"Chapter N").
3. Si nada coincide, la novela se trata como un unico capitulo (se avisa).
"""
import re
import unicodedata

from app.models.schemas import ChapterSpan, Paragraph

HEADING_KEYS = ("heading", "title", "titulo", "encabezado", "encabezamiento")

CHAPTER_RE = re.compile(
    r"^\s*(?:cap[ií]tulo|chapter)\s*[#:.\-]?\s*\d+",
    re.IGNORECASE,
)


def _fold(text: str) -> str:
    """Quita acentos y espacios para comparar nombres de estilo sin ambiguedad."""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.lower().replace(" ", "")


def is_heading_style(style: str) -> bool:
    return any(k in _fold(style) for k in HEADING_KEYS)


def detect_chapters(paragraphs: list[Paragraph]) -> list[ChapterSpan]:
    """Devuelve los capitulos como rangos de indices de parrafo (1-based)."""
    if not paragraphs:
        return []

    starts = [p.paragraph_index for p in paragraphs if is_heading_style(p.style)]

    if not starts:
        starts = [p.paragraph_index for p in paragraphs if CHAPTER_RE.match(p.text)]

    if not starts:
        return [
            ChapterSpan(
                chapter_index=1,
                title="Novela completa (sin capitulos detectados)",
                start_paragraph=0,
                end_paragraph=len(paragraphs) - 1,
            )
        ]

    spans: list[ChapterSpan] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] - 1 if i + 1 < len(starts) else len(paragraphs) - 1
        spans.append(
            ChapterSpan(
                chapter_index=i + 1,
                title=paragraphs[start].text,
                start_paragraph=start,
                end_paragraph=end,
            )
        )
    return spans
