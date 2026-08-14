# Memoria narrativa y fusión hybrid (Fases 2B y 2C)

Resultados de la segunda fuente de retrieval (índice narrativo de capítulos) y de su fusión
con el baseline de Qdrant. Datos brutos en `data/eval_answers/`; el informe completo de la
fusión en `data/eval_answers/hybrid_report.md`. Este documento queda commiteado para que la
experimentación y su conclusión sean trazables.

## Fase 2B — memoria narrativa (adoptada)

Índice local de la novela que localiza qué capítulos/trozos responden a una pregunta a partir
de la estructura narrativa (capítulos, personajes, líneas temporales). **Solo localiza**: el
texto final de cada candidato se resuelve siempre en Qdrant (nunca se usa
`narrative_memory.json` como evidencia literal).

| Métrica | baseline | memoria narrativa |
|---|---:|---:|
| recall@8 | 0.750 | **0.906** |
| capítulos objetivo que el baseline perdía | — | recupera **4/7** |

## Fase 2C — fusión hybrid (descartada)

Fusionar la memoria narrativa con el baseline (`scripts/evaluate_hybrid.py`) **no aportó**:

- No añade recall sobre la memoria sola.
- No mejora los grades de las respuestas generadas.
- Duplica el tiempo de retrieval.

## Conclusión

La **memoria narrativa se adopta** como segunda fuente de retrieval (recall@8 0.75 → 0.906).
La **fusión hybrid se descarta**: `RETRIEVAL_HYBRID=off` por defecto y el código se conserva
solo como referencia (p. ej. para regenerar el informe). A partir de aquí el cuello de
botella del sistema es la **generación**, no el retrieval.

## Reproducción

```powershell
# 1) Construir la memoria de una novela (aliases opcionales)
python scripts/build_narrative_memory.py --book data/books/<tu-novela>.docx --aliases data/aliases.json
#    → data/narrative_memory.json

# 2) Evaluar la memoria contra las mismas preguntas del baseline
python scripts/evaluate_narrative_retrieval.py --eval-file data/eval_questions.json

# 3) Comparar baseline vs memoria vs fusión (experimento Fase 2C)
python scripts/evaluate_hybrid.py --eval-file data/eval_questions.json --out data/eval_answers/hybrid.json
python scripts/build_hybrid_report.py   # → data/eval_answers/hybrid_report.md
```
