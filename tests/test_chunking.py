from app.ingestion.chapters import detect_chapters
from app.ingestion.chunking import chunk_paragraphs
from app.models.schemas import ChapterSpan, Paragraph


def make_paragraphs(texts):
    pos = 0
    paras = []
    for i, text in enumerate(texts):
        paras.append(
            Paragraph(
                text=text,
                style="Normal",
                paragraph_index=i,
                global_position=pos,
            )
        )
        pos += len(text) + 1
    return paras


def one_chapter(paragraphs):
    return [
        ChapterSpan(
            chapter_index=1,
            title="Capítulo 1",
            start_paragraph=0,
            end_paragraph=len(paragraphs) - 1,
        )
    ]


def two_chapters(paragraphs, split):
    return [
        ChapterSpan(chapter_index=1, title="Capítulo 1", start_paragraph=0, end_paragraph=split - 1),
        ChapterSpan(chapter_index=2, title="Capítulo 2", start_paragraph=split, end_paragraph=len(paragraphs) - 1),
    ]


def test_chunks_are_whole_paragraphs():
    paragraphs = make_paragraphs([f"Párrafo {i} con algo de contenido." for i in range(10)])
    chunks = chunk_paragraphs(paragraphs, one_chapter(paragraphs), "libro", chunk_tokens=40, chunk_overlap=10)

    assert len(chunks) >= 3
    for chunk in chunks:
        expected = paragraphs[chunk.paragraph_start : chunk.paragraph_end + 1]
        assert chunk.text == "\n\n".join(p.text for p in expected)
        assert chunk.paragraph_indices == list(range(chunk.paragraph_start, chunk.paragraph_end + 1))


def test_chunks_are_ordered_and_contiguous():
    paragraphs = make_paragraphs([f"Párrafo {i} con algo de contenido." for i in range(20)])
    chapters = two_chapters(paragraphs, split=10)
    chunks = chunk_paragraphs(paragraphs, chapters, "libro", chunk_tokens=40, chunk_overlap=10)

    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    positions = [c.global_position for c in chunks]
    assert positions == sorted(positions)


def test_chapters_are_respected():
    paragraphs = make_paragraphs([f"Párrafo {i} con algo de contenido." for i in range(20)])
    chapters = two_chapters(paragraphs, split=10)
    chunks = chunk_paragraphs(paragraphs, chapters, "libro", chunk_tokens=40, chunk_overlap=10)

    assert {c.chapter_index for c in chunks} == {1, 2}
    for chunk in chunks:
        chapter = chapters[chunk.chapter_index - 1]
        assert chunk.paragraph_start >= chapter.start_paragraph
        assert chunk.paragraph_end <= chapter.end_paragraph


def test_overlap_repeats_paragraphs():
    paragraphs = make_paragraphs([f"Párrafo {i} con algo de contenido." for i in range(12)])
    chunks = chunk_paragraphs(paragraphs, one_chapter(paragraphs), "libro", chunk_tokens=35, chunk_overlap=15)

    repeats = 0
    for a, b in zip(chunks, chunks[1:]):
        shared = set(a.paragraph_indices) & set(b.paragraph_indices)
        if shared:
            repeats += 1
    assert repeats >= 1

    for a, b in zip(chunks, chunks[1:]):
        assert a.chapter_index == b.chapter_index
        assert a.global_position <= b.global_position


def test_single_oversized_paragraph_not_split():
    big = "Palabra " * 400  # ~3200 chars -> ~800 tokens
    paragraphs = make_paragraphs([big, "Párrafo corto."])
    chunks = chunk_paragraphs(paragraphs, one_chapter(paragraphs), "libro", chunk_tokens=100, chunk_overlap=10)

    assert chunks[0].paragraph_start == chunks[0].paragraph_end == 0
    assert chunks[0].text == big


def test_no_overlap_when_zero():
    paragraphs = make_paragraphs([f"Párrafo {i}." for i in range(8)])
    chunks = chunk_paragraphs(paragraphs, one_chapter(paragraphs), "libro", chunk_tokens=30, chunk_overlap=0)
    for a, b in zip(chunks, chunks[1:]):
        assert not (set(a.paragraph_indices) & set(b.paragraph_indices))


def test_metadata_complete():
    paragraphs = make_paragraphs(["Párrafo uno.", "Párrafo dos.", "Párrafo tres."])
    chapters = detect_chapters(paragraphs)
    chunks = chunk_paragraphs(paragraphs, chapters, "mi-libro", chunk_tokens=500, chunk_overlap=50)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.book_id == "mi-libro"
    assert chunk.chapter_index == 1
    assert chunk.chapter_title == "Novela completa (sin capitulos detectados)"
    assert chunk.chunk_index == 0
    assert chunk.paragraph_start == 0
    assert chunk.paragraph_end == 2
    assert chunk.paragraph_indices == [0, 1, 2]
    assert chunk.global_position == 0
    assert chunk.characters == []
    assert chunk.text
