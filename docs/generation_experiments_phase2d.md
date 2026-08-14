# Prompt de grounding en la generación (Fase 2D) — descartado

**Veredicto: el prompt estricto de grounding se descarta.** Con modelo y contexto fijos
(`qwen3:1.7b`, `think=false`, mismo contexto recuperado que el baseline), reforzar el system
prompt para identificar primero la evidencia, distinguir hechos de inferencias, prohibir
conocimiento externo y citar fragmentos **no mejora** la calidad de las respuestas (incluso
empeoró ligeramente) y solo redujo tiempo y tokens.

Este documento queda commiteado para que la experimentación y su conclusión sean trazables.
Datos brutos e informe cruzado en `data/eval_answers/` (`compare_grounding_vs_baseline.md`).

## Hipótesis

Las alucinaciones del baseline podían deberse al prompt, no solo al modelo: un prompt que
obligue a separar evidencia de inferencia y prohíba el conocimiento externo debería reducir
las respuestas inventadas sin coste de velocidad.

## Metodología

- **Modelo y contexto fijos**: se regeneran las 16 respuestas sobre el contexto exacto del
  baseline (`evaluate_answers.py --from-results`), sin re-recuperar.
- **Prompt**: preset `--system-prompt grounding` frente al `baseline` (el de producción).
- **Rúbrica**: la de la Fase 2A (`correcta`, `parcial`, `incorrecta`, `sin_evidencia`,
  `alucinación`).

## Resultados (16 preguntas, `qwen3:1.7b` / `think=false`)

| Prompt | correcta | parcial | incorrecta | sin_evidencia | alucinación |
|---|---:|---:|---:|---:|---:|
| baseline | **3** | 8 | 4 | 0 | 1 |
| grounding | 2 | **9** | 3 | 0 | 2 |

El grounding solo redujo ligeramente el tiempo y los tokens por respuesta.

## Análisis

- La mejora esperada en alucinaciones **no aparece**: incluso aumentan (1 → 2).
- Los errores persisten **aunque la evidencia correcta esté en el contexto recuperado**: el
  modelo sigue equivocándose en varias preguntas con la respuesta a la vista.
- Ante huecos de retrieval, el modelo sigue **alucinando en lugar de declarar evidencia
  insuficiente** — el objetivo explícito del prompt grounding.

## Conclusión

El prompt no es el cuello de botella: lo es la **capacidad del modelo**. Se descarta el preset
`grounding` como opción de producción (se conserva solo como infraestructura de experimento,
`--system-prompt grounding`, sin cambios de comportamiento en runtime por defecto). El paso
siguiente natural es probar otros modelos con contexto y prompt fijos:
[benchmark de modelos (Fase 3)](model_benchmark_phase3.md).

## Reproducción

```powershell
# Regenerar las respuestas con el prompt "grounding" sobre el contexto del baseline
python scripts/evaluate_answers.py --from-results data/eval_answers/results_baseline700.json `
  --model qwen3:1.7b --system-prompt grounding --label grounding700

# Comparar baseline vs grounding
python scripts/compare_answers.py `
  --a-results data/eval_answers/results_baseline700.json --a-grades data/eval_answers/grades_baseline700.json --a-label baseline `
  --b-results data/eval_answers/results_grounding700.json --b-grades data/eval_answers/grades_grounding700.json --b-label grounding `
  --out data/eval_answers/compare_grounding_vs_baseline.md
```
