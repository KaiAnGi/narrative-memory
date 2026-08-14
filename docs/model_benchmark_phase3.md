# Benchmark de modelos de generación (Fase 3)

Resultados de probar otros modelos LLM con contexto, retrieval y prompt congelados: las
mismas 16 preguntas, el mismo contexto recuperado que el baseline, `--system-prompt baseline`
(el de producción) y sin `think`. Datos brutos y calificaciones en `data/eval_answers/`
(`results_*.json`, `grades_*.json`); este documento queda commiteado para que la conclusión
sea trazable.

## Metodología

- **Workload**: las mismas 16 preguntas de `data/eval_questions.json`.
- **Contexto congelado**: cada pregunta se regenera solo con
  `evaluate_answers.py --from-results` sobre el contexto exacto de `results_baseline700.json`
  (no se vuelve a recuperar).
- **Prompt**: `--system-prompt baseline`, sin `think`.
- **Rúbrica**: la de la Fase 2A (`correcta`, `parcial`, `incorrecta`, `sin_evidencia`,
  `alucinación`).
- **Hardware**: GTX 1050 Ti 4 GB (~2 GB VRAM libres) + 16 GB RAM.

## Resultados (16 preguntas, mismo contexto/retrieval/prompt)

| Modelo | correcta | parcial | incorrecta | sin_evidencia | alucinación | gen media | tok medio |
|---|---|---|---|---|---|---|---|
| `qwen3:1.7b` (baseline) | 3 | 8 | 4 | 0 | 1 | 16.2 s | 336 |
| `gemma2:2b` | **4** | 8 | **2** | 2 | **0** | 13.8 s | 192 |
| `phi3:mini` | 1 | 9 | 1 | 0 | **5** | 71.2 s | 605 |
| `llama3.2:3b` | 2 | 7 | 3 | 3 | 1 | 19.7 s | 213 |

Junto al benchmark controlado de la familia Qwen (mismo workload):
`qwen2.5:3b` ~18.1 s sin mejora global (descartado); `qwen3:4b` ~242-354 s y
`qwen3:8b` ~96-224 s (demasiado lentos para uso interactivo).

## Análisis

- **`gemma2:2b`**: única alternativa con mejora de calidad — más respuestas correctas, menos
  incorrectas y **cero alucinaciones**, a igual velocidad y con la mitad de tokens. A cambio,
  **2 negativas a responder** (declara no tener evidencia donde el contexto no bastaba), algo
  que el baseline sí intenta responder (a veces mal).
- **`phi3:mini`**: inutilizable en este hardware: ~71 s por respuesta (4.4× el baseline) y
  **5 alucinaciones**.
- **`llama3.2:3b`**: más lento que el baseline y peor calidad (menos respuestas correctas,
  3 negativas, respuestas erróneas).

## Conclusión

Ningún candidato local supera globalmente a `qwen3:1.7b` en calidad + velocidad. **Se mantiene
`qwen3:1.7b` como modelo por defecto.** `gemma2:2b` es una alternativa viable si se prioriza
evitar alucinaciones y aceptar negativas a responder. El cuello de botella es la **capacidad
del modelo** disponible en el hardware de 4 GB de VRAM, no el prompt ni el modelo concreto.

## Reproducción

```powershell
# 1) Regenerar las respuestas con cada modelo sobre el contexto congelado del baseline
python scripts/evaluate_answers.py --from-results data/eval_answers/results_baseline700.json `
  --model gemma2:2b --label gemma2_2b

# 2) Calificar (rellenar grades_<modelo>.json con la rúbrica Fase 2A)

# 3) Comparar cada modelo contra el baseline
python scripts/compare_answers.py `
  --a-results data/eval_answers/results_baseline700.json --a-grades data/eval_answers/grades_baseline700.json --a-label "qwen3:1.7b (baseline)" `
  --b-results data/eval_answers/results_gemma2_2b_<ts>.json --b-grades data/eval_answers/grades_gemma2_2b.json --b-label gemma2:2b `
  --out data/eval_answers/compare_gemma2_vs_baseline.md
```

Informes por modelo generados: `data/eval_answers/compare_{gemma2,phi3,llama3}_vs_baseline.md`.
