"""Modelos de datos compartidos por ingestion, retrieval y API."""
from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    """Un parrafo no vacio de la novela, con su posicion narrativa."""

    text: str
    style: str = "Normal"
    paragraph_index: int = 0
    global_position: int = 0  # offset acumulado de caracteres -> orden narrativo


class ChapterSpan(BaseModel):
    """Un capitulo como rango de indices de parrafo (end inclusive)."""

    chapter_index: int = 1  # 1-based
    title: str
    start_paragraph: int
    end_paragraph: int


class Chunk(BaseModel):
    """Fragmento indexado en Qdrant. Trazabilidad completa (ajuste #4)."""

    book_id: str
    chapter_index: int
    chapter_title: str
    chunk_index: int  # 0-based, global dentro del libro
    paragraph_start: int
    paragraph_end: int
    paragraph_indices: list[int] = Field(default_factory=list)
    global_position: int
    characters: list[str] = Field(default_factory=list)  # vacio en V1 (ajuste #5)
    text: str

    def payload(self) -> dict:
        return self.model_dump()


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    # Vector del chunk: solo se rellena cuando el reranking lo necesita (MMR).
    vector: list[float] | None = None


class SearchResult(BaseModel):
    query: str
    # Consultas expandidas (multi-query). Una sola entrada = sin expansion.
    queries: list[str] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    answer: str
    hits: list[SearchHit] = Field(default_factory=list)
    used_chapters: list[int] = Field(default_factory=list)


class IngestReport(BaseModel):
    book_id: str
    path: str
    paragraphs: int
    chapters: int
    chunks: int
