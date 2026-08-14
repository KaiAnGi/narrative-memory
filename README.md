# narrative-memory

Sistema local de memoria y razonamiento narrativo para novelas. Carga una novela `.docx`,
la indexa con embeddings en Qdrant y responde preguntas recuperando información de
**cualquier punto del libro** (no solo un resumen ni los últimos capítulos).

> El texto original es la fuente de verdad. Nunca se sustituye la novela por un resumen.

## Arquitectura

```text
novela.docx
   │
   ▼
extracción de texto (python-docx)
   │
   ▼
detección de capítulos (estilos Heading + regex)
   │
   ▼
chunking (≈700 tokens, overlap, sin cortar párrafos)
   │
   ▼
embeddings (qwen3-embedding:0.6b vía Ollama HTTP)
   │
   ├─────────────────────────────────────────────┐
   ▼                                             ▼
Qdrant (embebido, sin Docker)          narrative_memory.json
fuente principal de retrieval          (segunda fuente, Fase 2B;
                                       se construye con
                                       scripts/build_narrative_memory.py)
   │                                             │
   ▼                                             │
retrieval baseline ◄─── narrativo opcional ──────┘
   │              (fusión de ambas = hybrid,
   │               DESCARTADO en Fase 2C; off por defecto)
   ▼
Qwen3 1.7B (vía Ollama)  →  evidencia → razonamiento → conclusión
```

Sin frameworks de IA (ni LangChain ni similares): la comunicación con Ollama es HTTP directo.
Sin agentes ni PostgreSQL: son fases posteriores (ver [Roadmap](#roadmap)).

**Dos fuentes de retrieval:** el baseline de Qdrant (principal, intacto) y la **memoria
narrativa** (índice local de capítulos que localiza los trozos relevantes por estructura
narrativa y resuelve el texto final siempre en Qdrant). En la Fase 2C se probó además una
fusión de ambas (`scripts/evaluate_hybrid.py`) y el resultado fue **descartarla**: no mejora
nada sobre la memoria sola y duplica el tiempo. El veredicto completo está en
`data/eval_answers/hybrid_report.md`.

## Requisitos

- **Windows** (instrucciones pensadas para PowerShell).
- **Python 3.12 o 3.13** instalado.
- **Ollama** corriendo (`http://localhost:11434`) con los modelos:
  - `qwen3:1.7b` (LLM, recomendado para el hardware de desarrollo probado)
  - `qwen3-embedding:0.6b` (embeddings)
- **Docker NO es necesario**: Qdrant corre embebido en el proceso mediante la
  implementación local del propio `qdrant-client` (sin binario extra).

### Modelo recomendado (hardware de desarrollo probado)

En una GPU de **4 GB de VRAM** se midieron los siguientes tiempos por respuesta
(sobre el libro de ejemplo, 8 fragmentos de contexto):

| Modelo | think | tiempo por respuesta | observación |
|---|---|---|---|
| `qwen3:1.7b` | `false` | **~20 s** | configurado por defecto en esta V1 |
| `qwen3:1.7b` | `true` | ~25-37 s | razona, pero puede alucinar capítulos |
| `qwen3:8b` | `false` | ~1.5-2 min | viable si hace falta más capacidad |
| `qwen3:8b` | `true` | ~3-4 min | demasiado lento para uso interactivo |
| `qwen3:4b` | cualquiera | ~4-6 min | NO recomendado: con think=false mezcla el razonamiento en la respuesta |

Por eso la configuración actual es **`qwen3:1.7b` + `LLM_THINK=false`**: el razonamiento
oculto (thinking) de los modelos `qwen3` duplica/triplica el tiempo por respuesta, y
`think=false` lo desactiva explícitamente. Con `LLM_MODEL` puedes volver a `qwen3:8b` u
otro modelo, y con `LLM_THINK=true` reactivar el razonamiento si la pregunta lo exige.

## Instalación

```powershell
# 1) Clonar / entrar en el proyecto
cd narrative-memory

# 2) Crear y activar el entorno virtual
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Configurar variables de entorno
Copy-Item .env.example .env
#   Edita .env si tus URLs/modelos difieren (OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL...)

# 5) Asegurar los modelos de Ollama
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:1.7b
#   opcional si quieres más capacidad (más lento): ollama pull qwen3:8b
```

## Colocar tu novela

`data/` está fuera de Git a propósito: contiene tus libros, la base vectorial y los
resultados de evaluación y **no debe subirse al repositorio**. Copia tu `.docx` a:

```text
data/books/<tu-novela>.docx
```

(La carpeta se crea automáticamente; también puedes generarla a mano. El `book_id` que se
usa en las búsquedas deriva del nombre del archivo, p. ej. `La-Novia-2026.docx` → `la-novia-2026`.)

> El `.docx` debe usar **estilos de encabezado de Word** (Título 1 / Heading 1) para los
> capítulos: es el método de detección más fiable. Si no los usa, el sistema intenta por
> regex `"Capítulo N"`/`"Chapter N"`; si tampoco, trata todo como un único capítulo.

## Uso

### 1. Indexar la novela

```powershell
python scripts/ingest.py data/books/<tu-novela>.docx --verbose
```

Produce: párrafos extraídos, capítulos detectados y chunks indexados en Qdrant.
Re-indexar el mismo libro es **idempotente**: borra y reescribe sus chunks.

### 2. Hacer preguntas

```powershell
python scripts/ask.py "¿Qué sabe el protagonista en el capítulo 25 que su compañero todavía desconoce?"
python scripts/ask.py "¿Cómo evoluciona la relación entre ambos del capítulo 8 al 20?" --top-k 12
python scripts/ask.py "Busca solo en el capítulo 6" --chapter 6
```

Por defecto muestra los **fragmentos recuperados** (capítulo + posición global + score)
y luego la respuesta. Usa `--no-hits` para ver solo la respuesta.

La respuesta sigue el patrón **Evidencia → Razonamiento → Conclusión**, citando capítulos.
El modelo puede inferir sobre el contexto recuperado, pero **tiene prohibido inventar
acontecimientos**; si la evidencia no basta, lo dice explícitamente.

### 2b. Memoria narrativa (segunda fuente de retrieval, Fase 2B)

Índice local de la novela que localiza qué capítulos/trozos responden a una pregunta a
partir de la estructura narrativa (capítulos, personajes, líneas temporales). **Solo
localiza**: el texto final de cada candidato se resuelve siempre en Qdrant (nunca se usa
`narrative_memory.json` como evidencia literal).

```powershell
# 1) Construir la memoria de tu novela (aliases del libro opcionales)
python scripts/build_narrative_memory.py --book data/books/<tu-novela>.docx --aliases data/aliases.json
#    → data/narrative_memory.json

# 2) Evaluar la memoria contra las mismas preguntas del baseline (Fase 2B)
python scripts/evaluate_narrative_retrieval.py --eval-file data/eval_questions.json

# 3) Comparar baseline vs memoria vs fusión (experimento Fase 2C)
python scripts/evaluate_hybrid.py --eval-file data/eval_questions.json --out data/eval_answers/hybrid.json
python scripts/build_hybrid_report.py          # → data/eval_answers/hybrid_report.md
```

> **Veredicto Fase 2C (con datos):** la memoria narrativa mejora el retrieval del baseline
> (recall@8 de 0.75 → 0.906, recupera 4/7 capítulos que el baseline perdía). La **fusión
> hybrid está descartada**: no añade recall, no mejora los grades y duplica el tiempo de
> retrieval. El cuello de botella actual es la **generación** (`qwen3:1.7b`), no el
> retrieval: las respuestas correctas no subieron y aparecen más alucinaciones.

### 3. Evaluar la recuperación

Los tests unitarios verifican que el código funciona; para medir la **calidad** de la
recuperación crea un JSON y ejecuta:

```jsonc
// data/eval_questions.json
{
  "book_id": "la-novia-2026",
  "top_k": 8,
  "questions": [
    { "question": "¿Estaba ya enamorado X en el capítulo 20?", "expected_chapters": [20, 21] },
    { "question": "¿Qué pista del capítulo 6 adquiere sentido tras el 22?", "expected_chapters": [6, 22] }
  ]
}
```

```powershell
python scripts/evaluate_retrieval.py --eval-file data/eval_questions.json
```

Informa por pregunta qué capítulos recuperó vs. cuáles se esperaban (`recall@k`) y un
resumen agregado. El detalle se guarda en `data/eval_questions.results.json`.

### 3b. Experimentos de retrieval (Fase 1.5)

Compara **estrategias de recuperación** × **tamaños de chunk** con las mismas preguntas:

- Estrategias: `baseline` (búsqueda única), `multi-query` (expansión heurística de la
  pregunta en varias consultas), `multi-query+mmr` (multi-query + rerank MMR que reparte
  los resultados entre capítulos distintos).
- Chunk sizes: 300/50, 500/50, 700/50 y 700/100 tokens de tamaño/overlap, cada uno en su
  propia colección Qdrant (`narrative_c300_o50`, `narrative_c500_o50`,
  `narrative_c700_o100`, ...). El ganador fue **700/100**; la comparación completa está en
  `docs/retrieval_experiments_phase1_5.md`.

```powershell
python scripts/evaluate_retrieval_experiments.py --book data/books/<tu-novela>.docx
```

Reutiliza colecciones ya pobladas automáticamente (no re-embebe por defecto). Mide
`recall@k` (5/8/10), tasa de acierto (≥1 capítulo esperado), recall en preguntas
multi-capítulo y el tiempo medio por consulta. Salidas en `data/experiments/`
(detalle por configuración + `summary.json`); el baseline V1 no se toca.

### 4. API (opcional en V1)

```powershell
uvicorn app.api.main:app --reload
#   POST /ingest  {"path": "data/books/novela.docx"}
#   POST /search  {"query": "...", "top_k": 8, "chapter": null, "book_id": null}
#   POST /ask     {"question": "...", "top_k": 8, "chapter": null}
```

### Libro de ejemplo

Para probar sin tu novela:

```powershell
python scripts/make_sample_book.py
python scripts/ingest.py data/books/sample.docx
python scripts/ask.py "¿Qué pasa en el capítulo 3?"
```

## Configuración (`.env`)

| Variable | Por defecto | Descripción |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Servidor Ollama |
| `LLM_MODEL` | `qwen3:1.7b` | Modelo de chat (intercambiable: `qwen3:8b`, otros) |
| `LLM_THINK` | `false` | Raz. oculto de qwen3. `false` = respuesta directa (recomendado); `true` = razona primero (más lento) |
| `EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Modelo de embeddings (independiente del LLM) |
| `QDRANT_MODE` | `local` | `local` (embebido) o `remote` (servidor en `QDRANT_URL`) |
| `QDRANT_URL` | `http://localhost:6333` | Solo usado en modo `remote` |
| `QDRANT_LOCAL_PATH` | `data/qdrant_local` | Persistencia del Qdrant embebido (modo local) |
| `COLLECTION_NAME` | `narrative_chunks` | Nombre de la colección Qdrant |
| `BOOKS_DIR` | `data/books` | Carpeta con los `.docx` |
| `CHUNK_TOKENS` | `700` | Tamaño de chunk (700/100 recomendado por los experimentos Fase 1.5) |
| `CHUNK_OVERLAP` | `100` | Tokens de solapamiento entre chunks |
| `TOP_K` | `8` | Fragmentos recuperados por pregunta |
| `EMBEDDING_BATCH_SIZE` | `32` | Lote de textos por llamada a Ollama |
| `LLM_TEMPERATURE` | `0.2` | Temperatura del LLM |
| `RETRIEVAL_QUERY_EXPANSION` | `off` | Expansión multi-query: `off`, `heuristic` o `llm` (off ganó los experimentos; la LLM es mucho más lenta) |
| `RETRIEVAL_RERANK` | `none` | Rerank: `none` o `mmr` (diversidad por capítulo; none ganó los experimentos) |
| `RETRIEVAL_MAX_QUERIES` | `4` | Máximo de sub-consultas generadas por pregunta |
| `RETRIEVAL_CANDIDATES_PER_QUERY` | `8` | Candidatos por sub-consulta antes de fusionar/rerank |
| `RETRIEVAL_DIVERSITY_LAMBDA` | `0.7` | Peso de la diversidad en MMR (1 = solo relevancia) |
| `RETRIEVAL_CHAPTER_PENALTY` | `0.5` | Penalización a capítulos ya representados en MMR |
| `HYBRID_MEMORY_PATH` | `data/narrative_memory.json` | Memoria narrativa (segunda fuente, Fase 2B) |
| `RETRIEVAL_HYBRID` | `off` | **Experimental, DESCARTADO (Fase 2C).** `off` por defecto; la fusión Qdrant+memoria no aportó mejoras y se conserva solo como referencia (ver `data/eval_answers/hybrid_report.md`) |

Las variables `HYBRID_*` restantes (`HYBRID_NARRATIVE_TOP`, `HYBRID_CHUNKS_PER_CHAPTER`,
`HYBRID_FUSION`, `HYBRID_WEIGHT_BASELINE`, `HYBRID_WEIGHT_NARRATIVE`) solo se usan al
regenerar el informe del experimento. Los modelos se cambian con variables de entorno: el
sistema no está acoplado a ninguno.

## Tests

```powershell
.\.venv\Scripts\Activate.ps1
pytest            # suite offline (Ollama y Qdrant falsos)
pytest -m integration   # smoke test con Ollama real (requiere modelos instalados)
```

Los tests generan `.docx` en memoria y no necesitan red ni modelos.

## Estructura del proyecto

```text
app/
  core/          config centralizada (.env)
  models/        schemas compartidos (Chunk, SearchHit, Answer...)
  ingestion/     extractor, detección de capítulos, chunking
  embeddings/    OllamaEmbedder (interfaz Embedder)
  vector_store/  QdrantStore (local o remote, mismo interfaz)
  retrieval/     Searcher multi-query + expanders (off/heurístico/LLM) + reranker MMR
                 + hybrid.py (fusión experimental, DESCARTADO en Fase 2C)
  memory/        memoria narrativa: construcción, retrieval por capítulos y
                 postproceso con aliases (segunda fuente, Fase 2B)
  llm/           OllamaLLM + prompts (interfaz LLM intercambiable)
  api/           FastAPI mínima
  service.py     orquestación: ingest_book() / search() / ask_question()
scripts/         ingest.py, ask.py, make_sample_book.py,
                 build_narrative_memory.py, evaluate_narrative_retrieval.py,
                 evaluate_retrieval.py, evaluate_retrieval_experiments.py,
                 evaluate_hybrid.py, build_hybrid_report.py
tests/           unitarios + smoke de integración opcional
data/            (fuera de Git) books/, qdrant_local/, eval_answers/,
                 narrative_memory.json, aliases.json (datos de tu libro)
```

## Trazabilidad de cada fragmento

Cada chunk conserva: `book_id`, `chapter_index`, `chapter_title`, `chunk_index`,
`paragraph_start`, `paragraph_end`, `paragraph_indices`, `global_position` (offset de
caracteres = orden narrativo exacto), `characters` (vacío en V1) y el **texto original**.
Los chunks nunca cortan un párrafo a la mitad.

## Roadmap (fuera de la V1)

- **Fase 2B — memoria narrativa (completada):** índice de capítulos como segunda fuente de
  retrieval; mejora el recall@8 sobre el baseline y recupera 4/7 gaps que el baseline perdía.
- **Fase 2C — fusion hybrid (completada, descartada):** la fusion Qdrant+memoria no mejoro
  nada sobre la memoria sola y duplico el tiempo; se conserva solo como referencia
  (`retrieval_hybrid=off`). Veredicto: el cuello de botella es la **generacion**, no el
  retrieval.
- **Fase 3 (siguiente):** generacion con modelo mas capaz, grounding estricto en el contexto
  recuperado e instrucciones anti-alucinacion.
- Fase 4: agente con herramientas explicitas (`search_book`, `get_chapter`, ...).
- Fase 5: memoria narrativa estructurada en grafo (personajes, acontecimientos, relaciones,
  estado de conocimiento por personaje, cronologia) con PostgreSQL opcional.
- Extracción automática de personajes.
- Frontend Next.js + TypeScript.
