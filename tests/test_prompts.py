"""Tests de prompts de generacion (Fase 2D): variante grounding y resolucion."""
import pytest

from app.llm.prompts import (
    GROUNDING_SYSTEM_PROMPT,
    PROMPTS,
    SYSTEM_PROMPT,
    build_qa_messages,
    resolve_system_prompt,
)


def test_prompt_registry_has_baseline_and_grounding():
    assert "baseline" in PROMPTS
    assert "grounding" in PROMPTS
    assert PROMPTS["baseline"] is SYSTEM_PROMPT
    assert GROUNDING_SYSTEM_PROMPT != SYSTEM_PROMPT


def test_grounding_prompt_covers_strict_grounding_rules():
    for token in (
        "evidencia",
        "inferencia",
        "conocimiento externo",
        "fragmento",
        "cita",
    ):
        assert token in GROUNDING_SYSTEM_PROMPT.lower()


def test_resolve_by_name():
    assert resolve_system_prompt("baseline") == SYSTEM_PROMPT
    assert resolve_system_prompt("grounding") == GROUNDING_SYSTEM_PROMPT


def test_resolve_by_path(tmp_path):
    f = tmp_path / "prompt.txt"
    f.write_text("PROMPT PERSONALIZADO", encoding="utf-8")
    assert resolve_system_prompt(str(f)) == "PROMPT PERSONALIZADO"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_system_prompt("no-existe")


def test_build_qa_messages_uses_custom_system_prompt():
    messages = build_qa_messages(
        "Pregunta",
        hits=[],
        system_prompt=GROUNDING_SYSTEM_PROMPT,
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == GROUNDING_SYSTEM_PROMPT
    assert "Pregunta" in messages[1]["content"]


def test_build_qa_messages_defaults_to_baseline():
    messages = build_qa_messages("Pregunta", hits=[])
    assert messages[0]["content"] == SYSTEM_PROMPT
