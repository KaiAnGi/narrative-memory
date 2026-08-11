"""Pipeline RAG completo con Ollama y Qdrant falsos (corre sin red ni modelos)."""
import pytest


def test_ingest_report(service, make_docx):
    report = service.ingest_book(make_docx(n_chapters=3, paragraphs_per_chapter=4))

    assert report.book_id == "novela"
    assert report.chapters == 3
    assert report.paragraphs == 3 + 3 * 4
    assert report.chunks > 0
    assert service._store.count(report.book_id) == report.chunks


def test_reingest_is_idempotent(service, make_docx):
    path = make_docx(n_chapters=3, paragraphs_per_chapter=4)
    first = service.ingest_book(path)
    second = service.ingest_book(path)
    assert service._store.count(first.book_id) == second.chunks


def test_search_retrieves_expected_chapter(service, make_docx):
    service.ingest_book(make_docx(n_chapters=3, paragraphs_per_chapter=4))
    result = service.search("¿Qué ocurre en el capítulo 3?", top_k=8)
    assert result.hits
    assert result.hits[0].chunk.chapter_index == 3
    assert 3 in {h.chunk.chapter_index for h in result.hits}


def test_search_filter_by_chapter(service, make_docx):
    service.ingest_book(make_docx(n_chapters=3, paragraphs_per_chapter=4))
    result = service.search("pregunta cualquiera", top_k=8, chapter=2)
    assert result.hits
    assert all(h.chunk.chapter_index == 2 for h in result.hits)


def test_search_filter_by_book(service, make_docx):
    service.ingest_book(make_docx(n_chapters=2, paragraphs_per_chapter=3, name="otra.docx"))
    result = service.search("pregunta", top_k=8, book_id="otra")
    assert result.hits
    assert all(h.chunk.book_id == "otra" for h in result.hits)


def test_ask_question(service, make_docx):
    service.ingest_book(make_docx(n_chapters=3, paragraphs_per_chapter=4))
    answer = service.ask_question("¿Qué ocurre en el capítulo 2?", top_k=5)

    assert answer.answer == "RESPUESTA FAKE"
    assert answer.used_chapters
    assert answer.hits
    assert answer.question == "¿Qué ocurre en el capítulo 2?"


def test_embedder_dim_detected(service, make_docx):
    service.ingest_book(make_docx(n_chapters=1, paragraphs_per_chapter=2))
    assert service._embedder.dim() == 4


def test_missing_file_raises(service, tmp_path):
    with pytest.raises(FileNotFoundError):
        service.ingest_book(tmp_path / "no-existe.docx")
