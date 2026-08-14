# Experimentos del motor

Este directorio documenta los experimentos controlados del motor y sus conclusiones. Los
datos brutos (resultados, calificaciones, informes generados) viven en `data/eval_answers/`
y `data/experiments/` (fuera de Git); estos documentos quedan commiteados para que la
trazabilidad no dependa de `data/`.

| Fase | Tema | Veredicto | Documento |
|---|---|---|---|
| 1.5 | Estrategias de retrieval × tamaño de chunk | chunks **700/100**; multi-query y MMR descartados | [retrieval_experiments_phase1_5.md](retrieval_experiments_phase1_5.md) |
| 2A | Respuesta con evidencia: baseline `qwen3:1.7b` y evaluadores de respuestas | base de toda la evaluación de generación | — |
| 2B | Memoria narrativa (2ª fuente) y planner de consultas | memoria **adoptada** (recall@8 0.75 → 0.906); planner **descartado** | [memory_narrative_phase2b_2c.md](memory_narrative_phase2b_2c.md), [retrieval_experiments_phase2b.md](retrieval_experiments_phase2b.md) |
| 2C | Fusión hybrid Qdrant + memoria | **descartada** (`RETRIEVAL_HYBRID=off`) | [memory_narrative_phase2b_2c.md](memory_narrative_phase2b_2c.md) |
| 2D | Prompt de grounding en la generación | **descartado** | [generation_experiments_phase2d.md](generation_experiments_phase2d.md) |
| 3 | Benchmark de modelos LLM de generación | se mantiene `qwen3:1.7b` | [model_benchmark_phase3.md](model_benchmark_phase3.md) |

## Conclusión transversal

El cuello de botella del motor es la **capacidad del modelo** en el hardware de desarrollo
(GPU de 4 GB de VRAM), no el prompt ni la estrategia de retrieval: las mejoras de retrieval
(Fase 2B) suben el recall pero no se traducen en respuestas correctas, y ningún prompt ni
modelo local disponible supera globalmente al baseline `qwen3:1.7b` en calidad + velocidad.
