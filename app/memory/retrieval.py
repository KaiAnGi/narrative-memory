"""Primitivas de recuperacion sobre la memoria narrativa.

La memoria NO es una fuente de texto: es un mecanismo de LOCALIZACION. Cada
unidad (summary / events / relationships / characters / locations) se asocia a
los chunks originales (``chunk_refs`` del capitulo / ``source_chunks`` de cada
unidad). El score de un capitulo es la maxima similitud ponderada entre la
pregunta y sus unidades; el texto final SIEMPRE se resuelve en Qdrant a partir
de esos indices.

Compartido por:
  - ``scripts/evaluate_narrative_retrieval.py`` (experimento memoria vs baseline)
  - ``app/retrieval/hybrid.py`` (estrategia hybrid experimental)
"""
import json
import time
from pathlib import Path

WEIGHTS = {
    "summary": 1.0,
    "event": 1.0,
    "relationship": 0.9,
    "characters": 0.7,
    "locations": 0.7,
}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def build_units(memory: dict) -> list[dict]:
    """Construye las unidades de evidencia por capitulo de la memoria."""
    units = []
    for chapter in memory["chapters"]:
        cap = chapter["chapter_index"]
        units.append({
            "chapter_index": cap,
            "kind": "summary",
            "text": chapter.get("summary", ""),
            "source_chunks": chapter.get("chunk_refs", []),
        })
        for ev in chapter.get("events", []):
            units.append({
                "chapter_index": cap,
                "kind": "event",
                "text": ev.get("text", ""),
                "source_chunks": ev.get("source_chunks", []),
            })
        for rel in chapter.get("relationships", []):
            units.append({
                "chapter_index": cap,
                "kind": "relationship",
                "text": rel.get("relation", ""),
                "source_chunks": rel.get("source_chunks", []),
            })
        chars = [c for c in chapter.get("characters", []) if c]
        if chars:
            units.append({
                "chapter_index": cap,
                "kind": "characters",
                "text": " | ".join(chars),
                "source_chunks": [],
            })
        locs = [l for l in chapter.get("locations", []) if l]
        if locs:
            units.append({
                "chapter_index": cap,
                "kind": "locations",
                "text": " | ".join(locs),
                "source_chunks": [],
            })
    return units


def embed_units(
    embedder,
    units: list[dict],
    cache_path: Path,
    use_cache: bool = True,
) -> list[dict]:
    """Embeede cada unidad; cache en disco para reutilizar entre ejecuciones."""
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if len(cached["units"]) == len(units) and all(
            u["text"] == cu["text"] for u, cu in zip(units, cached["units"])
        ):
            print(f"Embeddings de la memoria cargados del cache: {cache_path.name}")
            return cached["units"]

    t0 = time.perf_counter()
    texts = [u["text"] for u in units]
    embeddings = embedder.embed(texts)
    for u, vec in zip(units, embeddings):
        u["embedding"] = vec
    elapsed = time.perf_counter() - t0
    print(f"Embeddings de {len(units)} unidades en {elapsed:.1f}s")

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"units": units}, ensure_ascii=False), encoding="utf-8"
        )
    return units


def score_question(question: str, q_emb: list[float], units: list[dict]) -> list[dict]:
    """Puntua cada capitulo: mejor similitud ponderada sobre sus unidades.

    Devuelve, por capitulo, el score y la evidencia ganadora.
    """
    by_chapter: dict[int, dict] = {}
    for u in units:
        sim = _cosine(q_emb, u["embedding"]) * WEIGHTS[u["kind"]]
        entry = by_chapter.setdefault(
            u["chapter_index"],
            {"chapter_index": u["chapter_index"], "score": -1.0, "evidence": None},
        )
        if sim > entry["score"]:
            entry["score"] = round(sim, 4)
            entry["evidence"] = {
                "kind": u["kind"],
                "text": u["text"],
                "source_chunks": u.get("source_chunks", []),
                "cosine": round(sim / WEIGHTS[u["kind"]], 4),
            }
    ranked = sorted(by_chapter.values(), key=lambda c: c["score"], reverse=True)
    return ranked
