# Experimento de retrieval planificado (Fase 2B) — descartado

**Veredicto: el planner de consultas se descarta.** Tras un experimento
controlado contra el baseline, la planificacion minima (detector de
complejidad barato + 2-3 consultas dirigidas por momento, reparto por rondas)
**empeoro** el retrieval: perdio recall y acierto, elimino capítulos esperados
que el baseline recuperaba y duplico el tiempo por consulta. El codigo de la
Fase 2B (planner, planificacion en `options.py`/`searcher.py`, flags en
`.env`) **no se commiteo**: el repositorio queda en el estado Fase 2A
(baseline + evaluadores de respuestas).

Este documento queda commiteado para que la experimentacion y su conclusion
sean trazables.

## Hipotesis

En el baseline multi-query las sub-consultas se fusionan por score maximo, por
lo que el ruido de una consulta dominante puede expulsar chunks relevantes de
otro momento de la pregunta. La idea de la Fase 2B era: para preguntas
complejas (multi-momento), planificar 2-3 consultas dirigidas (una por momento)
y repartir el top-k por rondas entre ellas, garantizando representacion de cada
momento. Deteccion de complejidad sin LLM (coste ~0): solo se activa si la
pregunta pasa un detector barato; las simples mantienen el flujo baseline
intacto.

## Metodologia

- **Preguntas**: las mismas 16 de `data/eval_questions.json` (fuera de Git).
- **Coleccion**: `narrative_c700_o100` (chunks 700/100, ganadora de la Fase 1.5).
- **Top-k**: 8. Metricas con k=5, 8, 10 (recall de capítulos esperados, acierto
  con >=1 capítulo esperado, recall medio multi-capítulo).
- **Estrategias** comparadas:
  - `baseline`: `expansion=off`, `rerank=none`, `planning=off` (reproduce el
    baseline de la Fase 1.5).
  - `planned` : `expansion=off`, `rerank=none`, `planning=heuristic` (detector
    barato + 2-3 consultas dirigidas por rondas, solo en complejas).
- **Harness**: `scripts/evaluate_answers.py` con modo `--strategy` (retirado
  junto con el codigo de la fase); datos brutos en `data/eval_answers/`
  (fuera de Git).

## Resultados

### Recall y acierto (16 preguntas, top-8)

| Estrategia | recall@5 | recall@8 | recall@10 | any@8 | multi_recall@8 | s/consulta |
|---|---|---:|---:|---:|---:|---:|---:|
| **baseline** | **0.719** | **0.750** | **0.750** | **0.938** (15/16) | **0.70** | **~3.05** |
| planned | 0.625 | 0.719 | 0.719 | 0.875 (14/16) | 0.65 | ~4.56 |

### Deteccion de complejidad

El detector marco 7/16 preguntas como complejas (Q8, Q10, Q11, Q12, Q13, Q15,
Q16) y genero 2-3 consultas dirigidas por pregunta.

### Efecto sobre capítulos objetivo

Se reviso manualmente el efecto sobre 7 preguntas cuyos capítulos esperados
eran difíciles (los que el baseline no recuperaba siempre): Q5→cap. 19,
Q7→cap. 12, Q8→cap. 1, Q11→cap. 24, Q12→cap. 5, Q13→cap. 13, Q15→cap. 20.

| Pregunta | baseline | planned |
|---|---|---|
| Q5 (cap. 19) | no recuperado | no detectada como compleja, sin cambios |
| Q7 (cap. 12) | no recuperado | no detectada como compleja, sin cambios |
| Q8 (cap. 1) | no recuperado | no recuperado |
| Q11 (cap. 24) | recuperado | **perdido** |
| Q12 (cap. 5) | no recuperado | no recuperado |
| Q13 (cap. 13) | no recuperado | no recuperado |
| Q15 (cap. 20) | no recuperado | **recuperado** (unica mejora) |

Ademas, el planner introdujo **regresiones**:

- **Q10**: perdio el cap. 23 que el baseline recuperaba; la respuesta generada
  paso a ser incorrecta ("robo de libros" en lugar de "acusación de asesinato").
- **Q11**: perdio el cap. 8 y la respuesta quedo generica (no podía responder).

## Causa raiz del fracaso

1. **Sustitucion de la pregunta por sub-consultas cortas**: el planner planea
   consultas de 3-8 palabras reescritas que embeddean peor que la pregunta
   original completa, perdiendo el contexto semantico que el baseline conserva
   al incluir siempre la pregunta original como primera consulta.
2. **Reparto por rondas sin resguardo del score**: la fusion round-robin
   garantiza representacion por momento pero expulsa chunks relevantes de alto
   score cuando hay mas consultas que momentos utiles, y mezcla ruido de
   consultas que no deberian aportar al top-k.
3. **Detector de complejidad imperfecto**: preguntas difíciles (Q5, Q7) ni
   siquiera se activaron; el beneficio teorico nunca llego a los casos que lo
   necesitaban.

## Conclusion

La planificacion minima, en su forma actual, no mejora el retrieval: empeora
recall y acierto, rompe respuestas que el baseline resolvía y cuesta ~1.5 s
mas por consulta. Se **descarta**. El baseline (Fase 1.5, chunks 700/100)
sigue siendo la configuracion por defecto y la base de la Fase 2A
(evaluacion de respuestas con qwen3:1.7b).
