"""Tests de la API FastAPI con un servicio falso (sin tocar Qdrant)."""
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.models.schemas import Answer, IngestReport, SearchResult


class FakeService:
    def ask_question(self, question, top_k=None, book_id=None, chapter=None):
        return Answer(question=question, answer="respuesta", hits=[], used_chapters=[])

    def ingest_book(self, path):
        return IngestReport(book_id="b", path=path, paragraphs=1, chapters=1, chunks=1)

    def search(self, query, top_k=None, book_id=None, chapter=None):
        return SearchResult(query=query)


def _patch_service(monkeypatch, fake):
    monkeypatch.setattr(api_main, "get_service", lambda: fake)


def test_health():
    client = TestClient(api_main.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_endpoint(monkeypatch):
    _patch_service(monkeypatch, FakeService())
    client = TestClient(api_main.app)
    resp = client.post("/ask", json={"question": "¿Quién ganó?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "respuesta"
    assert body["question"] == "¿Quién ganó?"


def test_ingest_endpoint(monkeypatch):
    _patch_service(monkeypatch, FakeService())
    client = TestClient(api_main.app)
    resp = client.post("/ingest", json={"path": "novela.docx"})
    assert resp.status_code == 200
    assert resp.json()["book_id"] == "b"


def test_search_endpoint(monkeypatch):
    _patch_service(monkeypatch, FakeService())
    client = TestClient(api_main.app)
    resp = client.post("/search", json={"query": "algo"})
    assert resp.status_code == 200
    assert resp.json()["query"] == "algo"
