"""Extraccion de texto desde un .docx usando python-docx.

Se conserva el texto original completo (fuente de verdad). No se resume nada.
"""
import re
from pathlib import Path

from docx import Document

from app.models.schemas import Paragraph


def slugify(name: str) -> str:
    """Deriva un book_id estable a partir del nombre del archivo."""
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem or "book"


def extract_docx(path: str | Path) -> list[Paragraph]:
    """Lee un .docx y devuelve sus parrafos no vacios, en orden.

    - ``paragraph_index``: indice secuencial entre parrafos no vacios.
    - ``global_position``: offset acumulado de caracteres (orden narrativo).
    - ``style``: nombre del estilo Word (usado para detectar capitulos).
    """
    doc = Document(str(path))
    paragraphs: list[Paragraph] = []
    position = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        try:
            style = para.style.name or "Normal"
        except Exception:  # noqa: BLE001
            style = "Normal"
        paragraphs.append(
            Paragraph(
                text=text,
                style=style,
                paragraph_index=len(paragraphs),
                global_position=position,
            )
        )
        position += len(text) + 1
    return paragraphs
