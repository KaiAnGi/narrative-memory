"""API FastAPI minima para la V1 (CLI sigue siendo el camino principal)."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import get_settings
from app.service import Service

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    svc = _service
    if svc is not None:
        svc.close()


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)

_service: Service | None = None


def get_service() -> Service:
    global _service
    if _service is None:
        _service = Service(settings)
    return _service


class IngestRequest(BaseModel):
    path: str


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    book_id: str | None = None
    chapter: int | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None
    book_id: str | None = None
    chapter: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    return get_service().ingest_book(req.path).model_dump()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return get_service().ask_question(
        req.question,
        top_k=req.top_k,
        book_id=req.book_id,
        chapter=req.chapter,
    ).model_dump()


@app.post("/search")
def search(req: SearchRequest) -> dict:
    return get_service().search(
        req.query,
        top_k=req.top_k,
        book_id=req.book_id,
        chapter=req.chapter,
    ).model_dump()
