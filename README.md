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
embeddings (vía Ollama HTTP)
   │
   ├─────────────────────────────────────────────┐
   ▼                                             ▼
Qdrant (embebido, sin Docker)          narrative_memory.json
fuente principal de retrieval          (segunda fuente: índice narrativo
                                        que localiza capítulos/trozos;
                                        ver docs/memory_narrative_phase2b_2c.md)
   │                                             │
   ▼                                             │
retrieval ────── fusión narrativa ───────────────┘
   │            (hybrid DESCARTADO, off por
   │             defecto — ver docs/README.md)
   ▼
LLM (vía Ollama; qwen3:1.7b por defecto)
   →  evidencia → razonamiento → conclusión
```

Sin frameworks de IA (ni LangChain ni similares): la comunicación con Ollama es HTTP directo.
Sin agentes ni PostgreSQL: son fases posteriores (ver [Roadmap](#roadmap)).

**Dos fuentes de retrieval:** el baseline de Qdrant (principal, intacto) y la **memoria
narrativa** (índice local de capítulos que localiza los trozos relevantes por estructura
narrativa y resuelve el texto final siempre en Qdrant). Los experimentos controlados y sus
conclusiones están en [docs/](docs/README.md).

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
(sobre una novela de ejemplo, 8 fragmentos de contexto):

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
otro modelo, y con `LLM_THINK=true` reactivar el razonamiento si la pregunta lo exige. El
benchmark controlado de alternativas está en [docs/model_benchmark_phase3.md](docs/model_benchmark_phase3.md).

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
usa en las búsquedas deriva del nombre del archivo, p. ej. `Mi-Libro.docx` → `mi-libro`.)

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
python scripts/ask.py "¿Qué pista del capítulo 25 explica un giro posterior del protagonista?"
python scripts/ask.py "¿Cómo evoluciona la relación entre dos personajes del capítulo 8 al 20?" --top-k 12
python scripts/ask.py "Busca solo en el capítulo 6" --chapter 6
```

Por defecto muestra los **fragmentos recuperados** (capítulo + posición global + score)
y luego la respuesta. Usa `--no-hits` para ver solo la respuesta.

La respuesta sigue el patrón **Evidencia → Razonamiento → Conclusión**, citando capítulos.
El modelo puede inferir sobre el contexto recuperado, pero **tiene prohibido inventar
acontecimientos**; si la evidencia no basta, lo dice explícitamente.

### 2b. Memoria narrativa (segunda fuente de retrieval)

Índice local de la novela que localiza qué capítulos/trozos responden a una pregunta a
partir de la estructura narrativa (capítulos, personajes, líneas temporales). **Solo
localiza**: el texto final de cada candidato se resuelve siempre en Qdrant (nunca se usa
`narrative_memory.json` como evidencia literal).

```powershell
# 1) Construir la memoria de tu novela (aliases del libro opcionales)
python scripts/build_narrative_memory.py --book data/books/<tu-novela>.docx --aliases data/aliases.json
#    → data/narrative_memory.json
```

### 3. Evaluación y experimentos

Para evaluar la **recuperación**, crea un JSON con las preguntas y los capítulos esperados:

```jsonc
// data/eval_questions.json
{
  "book_id": "<tu-novela>",
  "top_k": 8,
  "questions": [
    { "question": "¿Qué pista aparece en el capítulo 3 y se resuelve en el 10?", "expected_chapters": [3, 10] },
    { "question": "¿Qué información conoce el protagonista al final que al principio no?", "expected_chapters": [1, 12] }
  ]
}
```

```powershell
# Evaluar la recuperación (recall@k por pregunta)
python scripts/evaluate_retrieval.py --eval-file data/eval_questions.json

# Experimentos de retrieval (estrategias × tamaños de chunk)
python scripts/evaluate_retrieval_experiments.py --book data/books/<tu-novela>.docx

# Regenerar respuestas sobre el contexto congelado de una ejecución previa
python scripts/evaluate_answers.py --from-results data/eval_answers/results_baseline700.json --model <otro-modelo> --label <etiqueta>

# Comparar dos ejecuciones (grade, tiempo y tokens por pregunta)
python scripts/compare_answers.py `
  --a-results <a.json> --a-grades <a-grades.json> --a-label <A> `
  --b-results <b.json> --b-grades <b-grades.json> --b-label <B> --out <informe.md>
```

El detalle de cada experimento y su conclusión está en [docs/](docs/README.md); los datos
brutos y los informes generados viven en `data/eval_answers/`.

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
| `RETRIEVAL_QUERY_EXPANSION` | `off` | Expansión multi-query: `off`, `heuristic` o `llm` (`off` por defecto; ver docs/) |
| `RETRIEVAL_RERANK` | `none` | Rerank: `none` o `mmr` (diversidad por capítulo; `none` por defecto; ver docs/) |
| `RETRIEVAL_MAX_QUERIES` | `4` | Máximo de sub-consultas generadas por pregunta |
| `RETRIEVAL_CANDIDATES_PER_QUERY` | `8` | Candidatos por sub-consulta antes de fusionar/rerank |
| `RETRIEVAL_DIVERSITY_LAMBDA` | `0.7` | Peso de la diversidad en MMR (1 = solo relevancia) |
| `RETRIEVAL_CHAPTER_PENALTY` | `0.5` | Penalización a capítulos ya representados en MMR |
| `HYBRID_MEMORY_PATH` | `data/narrative_memory.json` | Memoria narrativa (segunda fuente) |
| `RETRIEVAL_HYBRID` | `off` | **Experimental, DESCARTADO (Fase 2C).** Fusión Qdrant+memoria, solo como referencia (ver [docs](docs/memory_narrative_phase2b_2c.md)) |

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
                 + hybrid.py (fusión experimental, DESCARTADO)
  memory/        memoria narrativa: construcción, retrieval por capítulos y
                 postproceso con aliases (segunda fuente)
  llm/           OllamaLLM + prompts (interfaz LLM intercambiable)
  api/           FastAPI mínima
  service.py     orquestación: ingest_book() / search() / ask_question()
scripts/         ingest.py, ask.py, make_sample_book.py,
                 build_narrative_memory.py, evaluate_narrative_retrieval.py,
                 evaluate_retrieval.py, evaluate_retrieval_experiments.py,
                 evaluate_hybrid.py, build_hybrid_report.py
tests/           unitarios + smoke de integración opcional
docs/            experimentos y conclusiones (ver docs/README.md)
data/            (fuera de Git) books/, qdrant_local/, eval_answers/,
                 narrative_memory.json, aliases.json (datos de tus libros)
```

## Trazabilidad de cada fragmento

Cada chunk conserva: `book_id`, `chapter_index`, `chapter_title`, `chunk_index`,
`paragraph_start`, `paragraph_end`, `paragraph_indices`, `global_position` (offset de
caracteres = orden narrativo exacto), `characters` (vacío en V1) y el **texto original**.
Los chunks nunca cortan un párrafo a la mitad.

## Roadmap (fuera de la V1)

- **Fase 2B — memoria narrativa (completada):** segunda fuente de retrieval adoptada
  ([docs](docs/memory_narrative_phase2b_2c.md)).
- **Fase 2C — fusión hybrid (completada, descartada):** se conserva solo como referencia,
  `RETRIEVAL_HYBRID=off` ([docs](docs/memory_narrative_phase2b_2c.md)).
- **Fase 2D — prompt grounding (completada, descartada):** queda como infraestructura de
  experimento, `--system-prompt grounding` ([docs](docs/generation_experiments_phase2d.md)).
- **Fase 3 — benchmark de modelos LLM (completada):** se mantiene `qwen3:1.7b`
  ([docs](docs/model_benchmark_phase3.md)).
- **Fase 4 (siguiente):** agente con herramientas explícitas (`search_book`, `get_chapter`, ...).
- Fase 5: memoria narrativa estructurada en grafo (personajes, acontecimientos, relaciones,
  estado de conocimiento por personaje, cronología) con PostgreSQL opcional.
- Extracción automática de personajes.
- Frontend Next.js + TypeScript.
