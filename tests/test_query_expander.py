"""Tests del expansor de consultas (multi-query)."""
import pytest

from app.retrieval.query_expander import (
    HeuristicQueryExpander,
    LLMQueryExpander,
    OffExpander,
    build_expander,
)


def test_off_returns_only_original():
    expander = OffExpander()
    assert expander.expand("pregunta cualquiera") == ["pregunta cualquiera"]


def test_heuristic_splits_desde_hasta():
    expander = HeuristicQueryExpander(max_queries=4)
    queries = expander.expand(
        "¿Cómo evoluciona la relación de los protagonistas desde el regreso "
        "del museo hasta el capítulo de las consecuencias?"
    )
    assert queries[0].startswith("¿Cómo evoluciona")
    assert len(queries) <= 4
    joined = " | ".join(q.lower() for q in queries)
    assert "regreso" in joined
    assert "consecuencias" in joined
    assert "protagonistas" in joined


def test_heuristic_fragment_and_global_terms():
    expander = HeuristicQueryExpander(max_queries=5)
    queries = expander.expand(
        "¿Qué herramienta de entrenamiento proyecta escenarios de combate "
        "y termina alterando la realidad durante una práctica?"
    )
    assert queries[0] == (
        "¿Qué herramienta de entrenamiento proyecta escenarios de combate "
        "y termina alterando la realidad durante una práctica?"
    )
    joined = " | ".join(q.lower() for q in queries)
    assert "herramienta" in joined
    assert "realidad" in joined


def test_heuristic_dedupes_and_caps():
    expander = HeuristicQueryExpander(max_queries=3)
    queries = expander.expand("¿Qué ocurre en el capítulo 3?")
    assert len(queries) <= 3
    assert queries[0].startswith("¿Qué ocurre")


def test_heuristic_empty_question():
    expander = HeuristicQueryExpander()
    assert expander.expand("") == [""]


class _FakeLLM:
    def chat(self, messages):
        return "regreso del museo protagonistas\n- consecuencias protagonistas\n3. evolución relación protagonistas"


def test_llm_parses_lines_and_keeps_original():
    expander = LLMQueryExpander(_FakeLLM(), max_queries=4)
    queries = expander.expand("¿Cómo evoluciona la relación desde el museo hasta las consecuencias?")
    assert queries[0].startswith("¿Cómo evoluciona")
    assert queries[1:] == [
        "regreso del museo protagonistas",
        "consecuencias protagonistas",
        "evolución relación protagonistas",
    ]


def test_build_expander_modes():
    assert isinstance(build_expander("off"), OffExpander)
    assert isinstance(build_expander("heuristic"), HeuristicQueryExpander)
    assert isinstance(build_expander("heuristic", max_queries=5).name, str)
    assert isinstance(build_expander("", max_queries=3), OffExpander)
    with pytest.raises(ValueError):
        build_expander("llm")
    assert isinstance(build_expander("llm", llm=_FakeLLM()), LLMQueryExpander)
