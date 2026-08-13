# Experimentos de retrieval (Fase 1.5)

Resultados sobre la novela de prueba local (`data/`),
misma 16 preguntas de `data/eval_questions.json` que el baseline V1. Datos brutos y
detalles por pregunta en `data/experiments/`; este documento queda
commiteado para que la conclusión sea trazable.

## Metodología

- **Estrategias** (`app/retrieval/options.py`):
  - `baseline`: `expansion=off`, `rerank=none` (= V1).
  - `multi-query`: expansión **heurística** (split por conectores) + top-k por score.
  - `multi-query+mmr`: expansión heurística + rerank MMR (diversidad por capítulo,
    `λ=0.7`, `chapter_penalty=0.5`).
- **Chunk sizes** (tokens de tamaño / overlap), cada uno en su propia colección Qdrant:
  300/50, 500/50 (colección V1 `narrative_chunks`), 700/50, 700/100.
- **Métricas** por pregunta con `k=5, 8, 10`: recall de capítulos esperados, acierto
  con ≥1 capítulo esperado, y recall medio de las 10 preguntas multi-capítulo.
- **Tiempo**: segundos por consulta (embeddings de sub-consultas + búsquedas).

## Resultados

### Recall medio por configuración (estrategia ganadora: baseline)

| Chunk | chunks | recall@5 | recall@8 | recall@10 | any@5 | any@8 | multich@8 | s/consulta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **300/50** | 776 | 0.531 | 0.688 | 0.719 | 0.812 | 0.938 | 0.60 | 2.60 |
| **500/50** (V1) | 448 | 0.594 | 0.719 | 0.750 | 0.875 | 0.938 | 0.65 | 2.60 |
| **700/50** | 318 | 0.656 | 0.719 | 0.719 | 0.875 | 0.938 | 0.65 | 2.64 |
| **700/100** | 341 | **0.719** | **0.750** | **0.781** | **0.938** | 0.938 | **0.70** | 2.64 |

### Estrategias por chunk size (recall@8 / recall@5)

| Chunk | baseline | multi-query | multi-query+mmr |
|---|---|---|---|
| 300/50 | **0.688 / 0.531** | 0.625 / 0.500 | 0.656 / 0.500 |
| 500/50 (V1) | **0.719 / 0.594** | 0.625 / 0.594 | 0.625 / 0.594 |
| 700/50 | **0.719 / 0.656** | 0.594 / 0.594 | 0.594 / 0.594 |
| 700/100 | **0.750 / 0.719** | 0.656 / 0.594 | 0.625 / 0.594 |

En **todas** las configuraciones el baseline iguala o supera a multi-query y a
multi-query+mmr.

## Análisis de por qué multi-query y MMR no mejoran

1. **El tamaño de chunk es el factor dominante.** 700/100 gana en todas las métricas
   (recall@5 0.719 vs 0.594 de la V1; el efecto es monotónico 300→500→700). Con chunks
   más grandes hay menos chunks por capítulo, menos aglutinamiento interno y cada chunk
   conserva más contexto de la escena.

2. **Las sub-consultas heurísticas añaden ruido, no capítulos nuevos.** Ejemplo
   diagnóstico:
   - El query original sí recupera el capítulo deseado (rank 8, score 0.599).
   - La sub-consulta **no recupera
     ningún chunk del capítulo deseado** (misma brecha de vocabulario que el baseline) y
     aporta chunks de otros capítulos con scores hasta 0.65.
   - Al fusionar por score máximo, esos chunks ruidosos **expulsan al capítulo deseado** del
     top-8: recall@8 pasa de 1.0 a 0.0.

3. **MMR reparte los 8 huecos sobre un pool inflado por ruido.** Al ampliar el pool con
   sub-consultas, MMR elige chunks de más capítulos distintos y diluye la representación
   del capítulo esperado. En el caso sintético de `tests/test_multiquery_searcher.py` el
   pool sí contenía los capítulos esperados y MMR los recuperó; en la novela real el pool
   los pierde antes.

4. **Las brechas de vocabulario persisten en todas las configuraciones.** Varias preguntas siguen sin
   recuperar su capítulo con recall@8 = 0 en cualquier combinación: la expansión heurística
   reformula con las mismas palabras, y ninguna estrategia cierra la brecha léxica.

## Coste temporal adicional por consulta

| Estrategia | consultas embebidas | s/consulta | extra vs baseline |
|---|---:|---:|---:|
| baseline | 1 | ~2.6 | — |
| multi-query (heurística) | 2.44 | ~6.4 | +~3.8 s (~2.5×) |
| multi-query + LLM (`qwen3:1.7b`, think=false) | 4 | ~6.5 + expansión | +~4–15 s adicionales |

Medición real de la expansión LLM (2 preguntas, modelo ya cargado): **15.5 s** la
primera (incluye warmup) y **3.4 s** la segunda. La expansión LLM sí genera sinónimos
relevantes, pero su coste es prohibitivo frente al
beneficio no medido.

## Recomendaciones

1. **Adoptar chunking 700/100** como producción: `CHUNK_TOKENS=700`, `CHUNK_OVERLAP=100`,
   y re-ingerir (`python scripts/ingest.py data/books/<novela>.docx`). Es la única
   mejora con evidencia (recall@5 +0.13 sobre la V1) y sin coste por consulta.
2. **Dejar multi-query y MMR desactivados por defecto** (`expansion=off`, `rerank=none`):
   en este conjunto de evaluación no mejoran recall y triplican el tiempo. Se mantienen
   implementados y configurables por `.env` (útil para otros libros o si el pool contiene
   los capítulos esperados).
3. **No usar la expansión LLM en producción**: coste de ~4–15 s extra por pregunta con
   `qwen3:1.7b`. Queda disponible como opción `llm` para experimentos puntuales.
4. **Próximo paso más prometedor** (mismo problema de brecha de vocabulario que motivó
   esta fase): búsqueda híbrida (léxica + semántica) o un reranker que use el texto del
   capítulo, no solo los embeddings.

## Reproducción

```powershell
# Re-ingerir una configuración nueva (300/50, 700/100, ...) y evaluarla
python scripts/evaluate_retrieval_experiments.py --book data/books/<tu-novela>.docx
# Reutilizar la colección V1 para 500/50 y evaluar todo
python scripts/evaluate_retrieval_experiments.py --book data/books/<tu-novela>.docx --reuse-collection "500:50:narrative_chunks"
# Recalcular summary.json a partir de los detalles
python scripts/summarize_experiments.py
```
