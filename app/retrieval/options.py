"""Opciones de recuperacion (Fase 1.5).

Configura la estrategia de retrieval de forma declarativa para poder comparar
variantes en evaluaciones y controlar el coste temporal por consulta:

- ``expansion``: como se generan las consultas adicionales.
  - ``off``       -> solo la pregunta (comportamiento V1; el que mejor resultado dio
    en los experimentos de Fase 1.5, por eso es el valor por defecto).
  - ``heuristic`` -> sin LLM, coste ~0 (no mejoro recall en los experimentos).
  - ``llm``       -> usa el LLM configurado; solo si el coste extra es aceptable.
- ``rerank``:
  - ``none`` -> fusionar candidatos y quedarse con los top-k por similitud (defecto).
  - ``mmr``  -> reordenar con diversidad de chunks y de capitulos.
"""
from dataclasses import dataclass

from app.core.config import Settings


@dataclass(frozen=True)
class RetrievalOptions:
    expansion: str = "off"             # "off" | "heuristic" | "llm"
    rerank: str = "none"               # "none" | "mmr"
    max_queries: int = 4               # numero maximo de consultas (incl. original)
    candidates_per_query: int = 8      # hits por consulta antes de fusionar
    diversity_lambda: float = 0.7      # peso de relevancia en MMR (1-lambda = diversidad)
    chapter_penalty: float = 0.5       # factor de novedad si el capitulo ya esta representado

    @classmethod
    def from_settings(cls, settings: Settings) -> "RetrievalOptions":
        return cls(
            expansion=settings.retrieval_query_expansion,
            rerank=settings.retrieval_rerank,
            max_queries=settings.retrieval_max_queries,
            candidates_per_query=settings.retrieval_candidates_per_query,
            diversity_lambda=settings.retrieval_diversity_lambda,
            chapter_penalty=settings.retrieval_chapter_penalty,
        )
