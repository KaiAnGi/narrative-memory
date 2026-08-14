"""Prototipo de memoria narrativa (ingestion offline, fuera del pipeline).

Genera ``data/narrative_memory.json`` con informacion por capitulo de una
novela: resumen, personajes presentes, lugares, acontecimientos y relaciones,
conservando trazabilidad hacia el texto original (indices de capitulo, de chunk
y de parrafo).

Diseno (prototipo aislado, sin tocar retrieval ni pipeline):
- El texto se lee con los mismos modulos de ingestion (extractor, capitulos,
  chunking 700/100) que la coleccion del .env, para que los
  ``chunk_index``/``paragraph_*`` coincidan con lo indexado en Qdrant.
- Cada capitulo se divide en bloques de ~4 chunks (limite de contexto del
  modelo) etiquetados como ``[CHUNK <idx>]``; por bloque se extraen eventos,
  personajes, lugares y relaciones con citas a chunks de origen.
- Despues se consolida el capitulo (resumen breve + dedupe) con el modelo.
- Checkpoint por capitulo en ``data/narrative_memory_parts/`` para poder
  reanudar tras interrupciones; al terminar se ensambla el JSON final.

El mapa de alias (normalizacion de nombres de personaje) es dato del libro y se
pasa con ``--aliases`` (JSON); el mecanismo vive en ``app/memory/postprocess``.

Modelo: por defecto ``qwen3:8b`` (el mas capaz disponible localmente; tarea
offline de ingestion, el tiempo no es critico). Se registra en ``meta``.

Uso:
  python scripts/build_narrative_memory.py --book data/books/<tu-novela>.docx
  python scripts/build_narrative_memory.py --only-assemble
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.ingestion.chapters import detect_chapters  # noqa: E402
from app.ingestion.chunking import chunk_paragraphs, estimate_tokens  # noqa: E402
from app.ingestion.extractor import extract_docx, slugify  # noqa: E402
from app.llm.ollama_llm import OllamaLLM  # noqa: E402
from app.memory.postprocess import load_aliases, postprocess_memory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PARTS_DIR = ROOT / "data" / "narrative_memory_parts"
OUT = ROOT / "data" / "narrative_memory.json"

# Contexto por bloque: 4 chunks de 700 tokens + prompt ~ cabe en 8k del modelo.
MAX_CHUNKS_PER_BLOCK = 4
MAX_RETRIES = 2

_EXTRACT_SYSTEM = (
    "Eres un analista literario experto en extraer informacion narrativa. "
    "Se te entregan fragmentos de una novela, cada uno etiquetado como "
    "[CHUNK <indice>]. Extrae SOLO lo que el texto respalde, sin inventar. "
    "Responde EXCLUSIVAMENTE con un objeto JSON valido, sin texto fuera, "
    "con esta forma exacta:\n"
    '{"events": [{"text": "evento breve en una frase", "source_chunks": [1, 2]}], '
    '"characters": ["Nombre"], "locations": ["Lugar"], '
    '"relationships": [{"relation": "descripcion breve", "source_chunks": [1]}]}\n'
    "- events: acontecimientos, revelaciones o desarrollos con su chunk de origen "
    "(indices globales de la novela).\n"
    "- characters: SOLO los personajes que aparecen o se mencionan en el fragmento.\n"
    "- locations: SOLO lugares presentes en el fragmento.\n"
    "- relationships: solo cuando el fragmento muestre o avance una relacion "
    "relevante entre personajes; vacio si no aplica."
)

_EXTRACT_USER = (
    "Fragmentos (indices de chunk globales de la novela):\n\n{blocks}\n\n"
    "Devuelve el JSON con los eventos, personajes y lugares de ESTE fragmento."
)

_CONSOLIDATE_SYSTEM = (
    "Eres un analista literario. Recibes las extracciones parciales de un "
    "capitulo de una novela. Consolida en un objeto JSON valido, sin texto "
    "fuera, con esta forma exacta:\n"
    '{"summary": "resumen breve del capitulo (3-5 frases)", '
    '"characters": ["Nombre"], "locations": ["Lugar"], '
    '"events": [{"text": "...", "source_chunks": [1, 2]}], '
    '"relationships": [{"relation": "...", "source_chunks": [1]}]}\n'
    "- summary: el resumen mas informativo y fiel posible, cubriendo todo el capitulo.\n"
    "- characters: lista deduplicada de personajes PRESENTES en el capitulo "
    "(no menciones historicas); usa un nombre canonico por personaje.\n"
    "- locations: lista deduplicada de lugares del capitulo.\n"
    "- events: fusiona y deduplica los eventos parciales, respetando el orden "
    "cronologico del capitulo y conservando los source_chunks originales.\n"
    "- relationships: deduplica y consolida las relaciones del capitulo."
)

_CONSOLIDATE_USER = (
    "Extracciones parciales del capitulo:\n\n{partials}\n\n"
    "Devuelve el JSON consolidado del capitulo."
)


def _parse_json(text: str) -> dict:
    """Parseo tolerante: fence de markdown y recorte al primer/lastimo { }."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _ask_json(llm: OllamaLLM, system: str, user: str) -> dict:
    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            raw = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])
            return _parse_json(raw)
        except (ValueError, KeyError) as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(1.0)
    raise RuntimeError(f"El LLM no devolvio JSON valido tras {MAX_RETRIES + 1} intentos: {last}")


def _block_text(chunks: list, indices: list[int]) -> str:
    blocks = []
    for idx, chunk in zip(indices, chunks):
        blocks.append(f"[CHUNK {chunk.chunk_index}]\n{chunk.text}")
    return "\n\n".join(blocks)


def _extract_block(llm: OllamaLLM, chunks: list) -> dict:
    indices = [c.chunk_index for c in chunks]
    user = _EXTRACT_USER.format(blocks=_block_text(chunks, indices))
    return _ask_json(llm, _EXTRACT_SYSTEM, user)


def _consolidate_chapter(llm: OllamaLLM, chapter_index: int, partials: list[dict]) -> dict:
    payload = json.dumps(partials, ensure_ascii=False, indent=2)
    user = _CONSOLIDATE_USER.format(partials=payload)
    try:
        return _ask_json(llm, _CONSOLIDATE_SYSTEM, user)
    except RuntimeError:
        # Fallback sin LLM: concatenar extracciones para no perder el capitulo.
        events, chars, locs, rels = [], [], [], []
        for p in partials:
            events += p.get("events", [])
            for ch in p.get("characters", []):
                if ch not in chars:
                    chars.append(ch)
            for lo in p.get("locations", []):
                if lo not in locs:
                    locs.append(lo)
            rels += p.get("relationships", [])
        return {
            "summary": f"(consolidacion automatica de {len(partials)} bloques)",
            "characters": chars,
            "locations": locs,
            "events": events,
            "relationships": rels,
        }


def _chapter_entry(llm: OllamaLLM, chapter, chapter_chunks, paras_start) -> dict:
    partials: list[dict] = []
    for i in range(0, len(chapter_chunks), MAX_CHUNKS_PER_BLOCK):
        block = chapter_chunks[i : i + MAX_CHUNKS_PER_BLOCK]
        partial = _extract_block(llm, block)
        partials.append(partial)
        print(f"    bloque {i // MAX_CHUNKS_PER_BLOCK + 1}: "
              f"{len(partial.get('events', []))} eventos, "
              f"{len(partial.get('characters', []))} personajes")

    consolidated = _consolidate_chapter(llm, chapter.chapter_index, partials)
    chunk_refs = [c.chunk_index for c in chapter_chunks]
    return {
        "chapter_index": chapter.chapter_index,
        "title": chapter.title,
        "summary": consolidated.get("summary", ""),
        "characters": consolidated.get("characters", []),
        "locations": consolidated.get("locations", []),
        "events": consolidated.get("events", []),
        "relationships": consolidated.get("relationships", []),
        "paragraph_start": chapter.start_paragraph,
        "paragraph_end": chapter.end_paragraph,
        "chunk_refs": chunk_refs,
        "token_estimate": sum(
            estimate_tokens(p.text) for p in paras_start[chapter.start_paragraph : chapter.end_paragraph + 1]
        ),
    }


def _assemble(parts: list[Path], meta: dict, aliases) -> None:
    chapters = []
    for part in sorted(parts, key=lambda p: int(p.stem.split("_")[1])):
        chapters.append(json.loads(part.read_text(encoding="utf-8")))
    out = {"schema_version": "0.1", "meta": meta, "chapters": chapters}
    postprocess_memory(out, aliases)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nnarrative_memory.json: {len(chapters)} capitulos (post-procesados) -> {OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototipo de memoria narrativa")
    parser.add_argument("--book", required=True, help="Ruta al .docx de la novela")
    parser.add_argument("--aliases", default=str(ROOT / "data" / "aliases.json"),
                        help="JSON con el mapa de alias (alias -> nombre canonico) del libro")
    parser.add_argument("--model", default="qwen3:8b", help="Modelo Ollama para la extraccion")
    parser.add_argument("--collection", default=None,
                        help="Coleccion Qdrant cuyos indices de chunk deben coincidir "
                             "(por defecto la de .env)")
    parser.add_argument("--only-assemble", action="store_true",
                        help="No extraer: solo ensamblar narrative_memory.json desde los checkpoints")
    args = parser.parse_args()

    settings = get_settings()
    if args.collection:
        settings.collection_name = args.collection
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    aliases = load_aliases(Path(args.aliases)) if Path(args.aliases).exists() else {}

    if args.only_assemble:
        _assemble(sorted(PARTS_DIR.glob("part_*.json")),
                  meta={"assembled_only": True, "aliases": args.aliases}, aliases=aliases)
        return

    book_path = Path(args.book)
    if not book_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {book_path}")
    book_id = slugify(book_path.stem)

    paragraphs = extract_docx(book_path)
    chapters = detect_chapters(paragraphs)
    chunks = chunk_paragraphs(
        paragraphs, chapters, book_id,
        chunk_tokens=settings.chunk_tokens, chunk_overlap=settings.chunk_overlap,
    )
    print(f"Libro: {book_path.name}  |  {len(chapters)} capitulos, {len(chunks)} chunks")

    llm = OllamaLLM(
        base_url=settings.ollama_base_url,
        model=args.model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        think=False,
    )

    done = {int(p.stem.split("_")[1]) for p in PARTS_DIR.glob("part_*.json")}
    meta = {
        "book_id": book_id,
        "source_docx": str(book_path),
        "llm_model": args.model,
        "llm_think": False,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "collection_ref": settings.collection_name,
        "aliases": args.aliases,
        "strategy": "bloques de ~4 chunks -> extraccion LLM -> consolidacion por capitulo",
        "generated": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "schema_version": "0.1",
    }

    t_start = time.perf_counter()
    for chapter in chapters:
        idx = chapter.chapter_index
        part_path = PARTS_DIR / f"part_{idx:02d}.json"
        if idx in done:
            print(f"cap {idx:>2} [{chapter.title}]  -> checkpoint existente, omitido")
            continue
        chapter_chunks = [c for c in chunks if c.chapter_index == idx]
        print(f"cap {idx:>2} [{chapter.title}]  {len(chapter_chunks)} chunks")
        entry = _chapter_entry(llm, chapter, chapter_chunks, paragraphs)
        part_path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
        elapsed = time.perf_counter() - t_start
        print(f"    guardado {part_path.name}  ({elapsed:.0f}s acumulados)")

    _assemble(sorted(PARTS_DIR.glob("part_*.json")), meta=meta, aliases=aliases)
    print(f"Tiempo total: {time.perf_counter() - t_start:.0f}s")


if __name__ == "__main__":
    main()
