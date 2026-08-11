from app.ingestion.chapters import detect_chapters, is_heading_style
from app.ingestion.extractor import extract_docx
from app.models.schemas import Paragraph


def make_paragraphs(texts, styles=None):
    styles = styles or ["Normal"] * len(texts)
    pos = 0
    paras = []
    for i, (text, style) in enumerate(zip(texts, styles)):
        paras.append(
            Paragraph(
                text=text,
                style=style,
                paragraph_index=i,
                global_position=pos,
            )
        )
        pos += len(text) + 1
    return paras


def test_is_heading_style():
    assert is_heading_style("Heading 1")
    assert is_heading_style("Heading 2")
    assert is_heading_style("Título 1")
    assert is_heading_style("Title")
    assert is_heading_style("Encabezado 3")
    assert not is_heading_style("Normal")
    assert not is_heading_style("Quote")
    assert not is_heading_style("Body Text")


def test_detect_by_heading_styles(make_docx):
    path = make_docx(n_chapters=3, paragraphs_per_chapter=3)
    paragraphs = extract_docx(path)
    chapters = detect_chapters(paragraphs)

    assert len(chapters) == 3
    assert [c.chapter_index for c in chapters] == [1, 2, 3]
    assert [c.title for c in chapters] == ["Capítulo 1", "Capítulo 2", "Capítulo 3"]
    # El ultimo capitulo termina en el ultimo parrafo
    assert chapters[-1].end_paragraph == len(paragraphs) - 1
    # Los rangos cubren todo sin solaparse
    assert chapters[1].start_paragraph == chapters[0].end_paragraph + 1


def test_detect_fallback_by_regex():
    paragraphs = make_paragraphs(
        [
            "Capítulo 1",
            "Cuerpo del capítulo uno.",
            "Capítulo 2: El misterio",
            "Cuerpo del capítulo dos.",
        ]
    )
    chapters = detect_chapters(paragraphs)
    assert len(chapters) == 2
    assert chapters[0].title == "Capítulo 1"
    assert chapters[1].title == "Capítulo 2: El misterio"


def test_detect_fallback_single_chapter():
    paragraphs = make_paragraphs(["Párrafo uno.", "Párrafo dos."])
    chapters = detect_chapters(paragraphs)
    assert len(chapters) == 1
    assert chapters[0].chapter_index == 1
    assert chapters[0].start_paragraph == 0
    assert chapters[0].end_paragraph == 1


def test_detect_empty_book():
    assert detect_chapters([]) == []
