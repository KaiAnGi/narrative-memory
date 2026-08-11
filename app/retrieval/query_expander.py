"""Expansion de consultas (multi-query) sin agentes ni frameworks.

Estrategias intercambiables para generar varias consultas de busqueda a partir
de una pregunta compleja:

- ``HeuristicQueryExpander``: sin LLM. Divide la pregunta en fragmentos por
  conectores temporales/comparativos ("desde X hasta Y") y extrae terminos
  clave. Determinista y con coste ~0.
- ``LLMQueryExpander``: pide al LLM configurado que genere consultas de
  busqueda breves. Solo recomendado cuando el coste extra por consulta
  (una llamada al LLM) sea aceptable.
- ``OffExpander``: devuelve la pregunta tal cual (comportamiento V1).

El codigo es generico: no contiene reglas sobre capitulos ni libros concretos.
"""
import re
from typing import Protocol

from app.llm.ollama_llm import LLM

STOPWORDS = frozenset(
    """un una unos unas el la los las lo y o u ni de del que como cómo qué cuál
    cuáles cual cuales cuando donde dónde cuándo por para a ante con sin sobre
    entre hacia en es son era eran fue fueron sea ser esta este estos estas su
    sus tu tus mi mis nuestro nuestra al se me te le les nos os ya pero sino
    porque cuya cuyo quien quienes además durante aunque desde hasta entre
    después después de antes de había había sido""".split()
)

# Conectores que separan fragmentos con informacion parcial (p.ej. "X desde A
# hasta B" -> [X, A, B]). La pregunta original siempre es la primera consulta.
_SPLIT_RE = re.compile(
    r"\s+(?:desde|hasta|entre|pero|en cambio|mientras que|además de|a diferencia"
    r" de|al contrario que|después de|antes de)\s+",
    flags=re.IGNORECASE,
)

_QUESTION_WORD_RE = re.compile(
    r"^(?:¿|cuál|cuáles|qué|cómo|por qué|dónde|cuándo|quién|quiénes)\s*",
    flags=re.IGNORECASE,
)

_WORD_RE = re.compile(r"[a-záéíóúüñ]{3,}", flags=re.IGNORECASE)


class QueryExpander(Protocol):
    def expand(self, question: str) -> list[str]:
        ...


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        key = re.sub(r"\s+", " ", item).strip().lower()
        if key and key not in seen:
            seen.append(key)
            yield item


def _strip_question_word(question: str) -> str:
    return _QUESTION_WORD_RE.sub("", question.strip().strip("¿?")).strip()


def _key_terms(text: str, max_terms: int = 6) -> list[str]:
    tokens = [t.lower() for t in _WORD_RE.findall(text) if t.lower() not in STOPWORDS]
    terms: list[str] = []
    for t in tokens:
        if t not in terms:
            terms.append(t)
    return terms[:max_terms]


class OffExpander:
    """Devuelve la pregunta tal cual (baseline V1)."""

    def __init__(self) -> None:
        self.name = "off"

    def expand(self, question: str) -> list[str]:
        return [question.strip()]


class HeuristicQueryExpander:
    def __init__(self, max_queries: int = 4, min_terms: int = 2) -> None:
        self._max_queries = max(1, max_queries)
        self._min_terms = min_terms
        self.name = "heuristic"

    def expand(self, question: str) -> list[str]:
        original = question.strip()
        if not original:
            return [original]

        queries: list[str] = [original]
        cleaned = _strip_question_word(original)

        # 1) Fragmentos por conectores ("desde X hasta Y" -> consultas X e Y).
        for fragment in _SPLIT_RE.split(cleaned):
            terms = _key_terms(fragment)
            if len(terms) >= self._min_terms:
                queries.append(" ".join(terms))

        # 2) Terminos clave globales (respaldo para preguntas sin conectores).
        global_terms = _key_terms(cleaned)
        if len(global_terms) >= self._min_terms:
            queries.append(" ".join(global_terms))

        return list(_dedupe(queries))[: self._max_queries]


class LLMQueryExpander:
    def __init__(self, llm: LLM, max_queries: int = 4) -> None:
        self._llm = llm
        self._max_queries = max(1, max_queries)
        self.name = "llm"

    def expand(self, question: str) -> list[str]:
        prompt = (
            "Genera entre 2 y {n} consultas de búsqueda breves en español para "
            "encontrar fragmentos de una novela que ayuden a responder la "
            "pregunta. Cada consulta debe tener 3-8 palabras, contener solo "
            "términos de búsqueda (sin pregunta completa ni signos de "
            "puntuación) y cubrir una parte distinta de la pregunta. Escribe "
            "una consulta por línea, sin numerar y sin guiones.\n\n"
            "Pregunta: {q}".format(n=self._max_queries - 1, q=question)
        )
        raw = self._llm.chat([{"role": "user", "content": prompt}])
        queries: list[str] = [question.strip()]
        for line in raw.splitlines():
            line = re.sub(r"^[\s\-*\d\.\)]+", "", line).strip()
            line = re.sub(r"[¿?]+$", "", line).strip()
            if 3 <= len(line.split()) <= 12:
                queries.append(line)
        return list(_dedupe(queries))[: self._max_queries]


def build_expander(
    mode: str | None,
    max_queries: int = 4,
    llm: LLM | None = None,
) -> QueryExpander:
    """Fabrica el expansor segun el modo configurado."""
    mode = (mode or "off").strip().lower()
    if mode == "llm":
        if llm is None:
            raise ValueError("expansión 'llm' requiere una instancia de LLM")
        return LLMQueryExpander(llm, max_queries=max_queries)
    if mode == "heuristic":
        return HeuristicQueryExpander(max_queries=max_queries)
    return OffExpander()
