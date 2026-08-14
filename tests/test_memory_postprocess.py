"""Tests del post-procesado determinista y reversible de la memoria narrativa."""
import pytest

from app.memory.postprocess import (
    DEDUPE_JACCARD,
    canonical_name,
    dedupe_events,
    load_aliases,
    normalize_names,
    normalize_text,
    postprocess_memory,
)

# Mapa de alias de PRUEBA (neutral, no de ninguna novela concreta): el
# mecanismo es generico y el mapa real es dato de cada libro.
ALIASES = {
    "james moriarty": "James",
    "jim": "James",
    "john smith": "John",
    "sir edgar": "Edgar",
    "doc": "Edgar",
}


def _chapter(**overrides) -> dict:
    chapter = {
        "chapter_index": 1,
        "title": "Test",
        "summary": "Resumen intacto.",
        "characters": [],
        "locations": [],
        "events": [],
        "relationships": [],
        "source_chunks": [],
    }
    chapter.update(overrides)
    return chapter


# --- normalizacion de alias --------------------------------------------------


def test_canonical_name_maps_alias():
    assert canonical_name("James Moriarty", ALIASES) == "James"
    assert canonical_name("Jim", ALIASES) == "James"
    assert canonical_name("John Smith", ALIASES) == "John"


def test_canonical_name_without_aliases_is_identity():
    assert canonical_name("James Moriarty") == "James Moriarty"
    assert canonical_name("Griffin") == "Griffin"


def test_canonical_name_keeps_unknown_names():
    assert canonical_name("El hombre de cabello claro", ALIASES) == "El hombre de cabello claro"
    assert canonical_name("Los tres ladrones", ALIASES) == "Los tres ladrones"
    assert canonical_name("Griffin", ALIASES) == "Griffin"


def test_normalize_text_replaces_in_free_text():
    text = "El narrador habla con James Moriarty y luego con John Smith."
    assert normalize_text(text, ALIASES) == "El narrador habla con James y luego con John."


def test_normalize_text_does_not_break_contained_words():
    # "doc" no debe mapearse dentro de "doctor".
    assert normalize_text("El doctor sonrie", ALIASES) == "El doctor sonrie"
    # "doc" como palabra independiente si se mapea.
    assert normalize_text("doc asintio", ALIASES) == "Edgar asintio"


# --- Jim -> James (apodo canonico) -------------------------------------------


def test_jim_to_james_evidence_based():
    # El mecanismo normaliza el apodo hacia el canonico definido en el mapa.
    text = "El narrador le envia una carta a James con un apodo privado llamandolo 'Jim'."
    assert normalize_text(text, ALIASES) == (
        "El narrador le envia una carta a James con un apodo privado llamandolo 'James'."
    )


def test_alias_applied_to_characters_and_events():
    chapter = _chapter(
        characters=["James", "Jim", "Edgar"],
        events=[{"text": "James recibe una carta a nombre de Jim.", "source_chunks": [7]}],
    )
    normalize_names(chapter, ALIASES)
    assert "Jim" not in chapter["characters"]
    assert chapter["characters"].count("James") == 1
    assert chapter["events"][0]["text"] == "James recibe una carta a nombre de James."


# --- no fusionar personajes distintos ----------------------------------------


def test_no_merge_distinct_characters_by_similarity():
    # "James" y "James Moriarty" son el mismo (mapa), pero nombres con
    # parecido textual sin evidencia no deben fusionarse.
    chapter = _chapter(
        characters=["El hombre de cabello claro", "El hombre de los ojos azules"],
    )
    normalize_names(chapter, ALIASES)
    assert chapter["characters"] == [
        "El hombre de cabello claro",
        "El hombre de los ojos azules",
    ]


def test_no_merge_unknown_similar_names():
    # No inventar identidades: "El muchacho alado" != "el hombre alado".
    assert canonical_name("El muchacho alado", ALIASES) == "El muchacho alado"
    assert canonical_name("el hombre alado", ALIASES) == "el hombre alado"


def test_characters_raw_preserves_originals():
    chapter = _chapter(characters=["James Moriarty", "Jim", "James"])
    normalize_names(chapter, ALIASES)
    assert chapter["characters_raw"] == ["James Moriarty", "Jim", "James"]
    assert chapter["characters"] == ["James"]


def test_aliases_loadable_from_json_file(tmp_path):
    # El mapa de alias del libro es un dato (JSON) cargado por el mecanismo.
    p = tmp_path / "aliases.json"
    p.write_text('{"jim": "James", "james moriarty": "James"}', encoding="utf-8")
    assert load_aliases(p) == {"jim": "James", "james moriarty": "James"}


def test_aliases_loadable_with_metadata(tmp_path):
    p = tmp_path / "aliases.json"
    p.write_text(
        '{"aliases": {"jim": "James"}, "evidence": {"jim": "apodo de James"}}',
        encoding="utf-8",
    )
    assert load_aliases(p) == {"jim": "James"}


# --- deduplicacion de eventos -------------------------------------------------


def test_dedupe_merges_adjacent_near_identical_events():
    chapter = _chapter(
        events=[
            {"text": "William se dirige a la oficina de la Directora Faragonda.",
             "source_chunks": [127]},
            {"text": "William entra en la oficina de la Directora Faragonda y se congelo.",
             "source_chunks": [127]},
        ]
    )
    dedupe_events(chapter)
    assert len(chapter["events"]) == 1
    # Se conserva el texto mas informativo.
    assert "se congelo" in chapter["events"][0]["text"]


def test_dedupe_keeps_distinct_adjacent_events():
    chapter = _chapter(
        events=[
            {"text": "Sherlock le muestra a William una moto aerea y le ofrece un casco magico.",
             "source_chunks": [3]},
            {"text": "William monta en una moto aerea con Sherlock.",
             "source_chunks": [3]},
        ]
    )
    dedupe_events(chapter)
    assert len(chapter["events"]) == 2


def test_dedupe_merges_repeated_near_identical_chain():
    chapter = _chapter(
        events=[
            {"text": "William se dirige a la oficina de la Directora Faragonda.",
             "source_chunks": [127]},
            {"text": "William entra en la oficina de la Directora Faragonda y se congelo.",
             "source_chunks": [127]},
            {"text": "William entra a la oficina de Faragonda y se encuentra con Griffin.",
             "source_chunks": [128]},
        ]
    )
    dedupe_events(chapter)
    assert len(chapter["events"]) == 1


# --- conservacion de source_chunks -------------------------------------------


def test_dedupe_preserves_union_of_source_chunks():
    chapter = _chapter(
        events=[
            {"text": "Sherlock propone ir al Lago Rocaluz", "source_chunks": [230]},
            {"text": "William y Sherlock deciden ir al Lago Rocaluz", "source_chunks": [231]},
        ]
    )
    dedupe_events(chapter)
    assert chapter["events"][0]["source_chunks"] == [230, 231]


def test_dedupe_keeps_chunks_of_surviving_events():
    chapter = _chapter(
        events=[
            {"text": "William y Sherlock llegan al Lago Rocaluz y lo observan",
             "source_chunks": [234]},
            {"text": "William se disculpa por haber besado a Sherlock",
             "source_chunks": [242]},
        ]
    )
    dedupe_events(chapter)
    assert chapter["events"] == [
        {"text": "William y Sherlock llegan al Lago Rocaluz y lo observan",
         "source_chunks": [234]},
        {"text": "William se disculpa por haber besado a Sherlock",
         "source_chunks": [242]},
    ]


# --- summary intacto ----------------------------------------------------------


def test_postprocess_does_not_touch_summary_or_locations():
    chapter = _chapter(
        summary="James Moriarty visita el lago con Jim.",
        characters=["Jim"],
        events=[{"text": "James Moriarty pasea con Jim.", "source_chunks": [1]}],
    )
    normalize_names(chapter, ALIASES)
    assert chapter["summary"] == "James Moriarty visita el lago con Jim."
    assert chapter["locations"] == []


# --- idempotencia -------------------------------------------------------------


def test_normalize_names_is_idempotent():
    chapter = _chapter(
        characters=["James Moriarty", "Jim", "John Smith"],
        events=[{"text": "James Moriarty llama a Jim por su apodo.", "source_chunks": [7]}],
    )
    first = json_roundtrip(chapter)
    normalize_names(first, ALIASES)
    snapshot = json_roundtrip(first)
    normalize_names(first, ALIASES)
    assert first == snapshot


def test_dedupe_events_is_idempotent():
    chapter = _chapter(
        events=[
            {"text": "William se dirige a la oficina de la Directora Faragonda.",
             "source_chunks": [127]},
            {"text": "William entra en la oficina de la Directora Faragonda y se congelo.",
             "source_chunks": [127]},
        ]
    )
    dedupe_events(chapter)
    snapshot = json_roundtrip(chapter)
    dedupe_events(chapter)
    assert chapter == snapshot


def test_postprocess_memory_is_idempotent():
    data = {
        "schema_version": "0.1",
        "meta": {},
        "chapters": [
            _chapter(
                characters=["James Moriarty", "Jim"],
                events=[
                    {"text": "James Moriarty lee la carta a nombre de Jim.",
                     "source_chunks": [7]},
                    {"text": "James entra en la oficina del director.",
                     "source_chunks": [127]},
                ],
            )
        ],
    }
    first = json_roundtrip(postprocess_memory(data, ALIASES))
    second = json_roundtrip(postprocess_memory(json_roundtrip(first), ALIASES))
    assert second == first


# --- utilidades ---------------------------------------------------------------


def json_roundtrip(obj):
    import json

    return json.loads(json.dumps(obj, ensure_ascii=False))
