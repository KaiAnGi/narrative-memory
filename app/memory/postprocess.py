"""Post-procesado determinista y reversible de la memoria narrativa.

Se aplica al ensamblar ``narrative_memory.json`` desde los checkpoints en bruto
(``data/narrative_memory_parts/``). Los checkpoints nunca se modifican, de modo
que el JSON final puede regenerarse de forma reproducible y revertirse
re-ensamblando sin este paso.

Dos operaciones:
1. ``normalize_names``: unifica variantes de nombre de personaje a un nombre
   canonico mediante un mapa de alias. El mapa es dato especifico del libro
   (JSON, p. ej. ``data/aliases.json``) y se inyecta como parametro: este
   modulo solo aporta el MECANISMO. Solo aplica cuando hay evidencia textual
   (el mapa lo documenta); no inventa identidades ni fusiona por parecido
   textual.
2. ``dedupe_events``: fusiona unicamente eventos casi identicos ADYACENTES en
   la lista del capitulo (mismo instante narrativo), conservando la union de
   ``source_chunks`` y el texto mas informativo.

El proceso es idempotente: aplicar el post-procesado dos veces produce el
mismo resultado.
"""
import json
import re
import unicodedata
from pathlib import Path

# Umbral de Jaccard sobre tokens de contenido para considerar dos eventos
# "casi identicos". Calibrado sobre la memoria: 0.5 fusiona los pares
# adyacentes reales (cap. 10 y 16) y respeta eventos distintos (caps. 7, 8, 18).
DEDUPE_JACCARD = 0.5

_STOPWORDS = set(
    "un una unos unas el la los las lo y o u ni de del que como en por para a "
    "ante con sin sobre entre hacia es son era eran fue fueron sea ser esta "
    "este estos estas su sus tu tus mi mis nuestro nuestra al se me te le les "
    "nos os ya pero sino porque mientras cuando desde hasta si la no".split()
)


def load_aliases(path) -> dict[str, str]:
    """Carga el mapa de alias (alias -> canonico) desde un JSON.

    El archivo puede ser un dict plano {alias: canonico} o, si quieres anotar
    la evidencia, un dict con la clave ``"aliases"`` junto a metadatos.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "aliases" in data:
        return data["aliases"]
    return data


def _alias_matcher(aliases: dict[str, str]):
    """Compila el mapa de alias en (claves, regexes) para reemplazo.

    Orden: frases completas antes que apocopes, para no mapear "seb" dentro de
    "sebastian" ni cortar frases mas largas.
    """
    keys = sorted(aliases, key=len, reverse=True)
    regexes = [
        re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
        for alias in keys
    ]
    return keys, regexes


def canonical_name(name: str, aliases: dict[str, str] | None = None) -> str:
    """Devuelve el nombre canonico de un personaje (o el original si no hay alias)."""
    return (aliases or {}).get(name.strip().lower(), name)


def normalize_text(text: str, aliases: dict[str, str] | None = None) -> str:
    """Reemplaza las variantes de nombre por el canonico en un texto libre."""
    aliases = aliases or {}
    for alias, regex in zip(*_alias_matcher(aliases)):
        if regex.search(text):
            text = regex.sub(aliases[alias], text)
    return text


def _content_tokens(text: str) -> set[str]:
    """Tokens de contenido: minusculas, sin acentos, sin stopwords ni palabras cortas."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    words = re.findall(r"[a-z0-9]+", text)
    return {w for w in words if w not in _STOPWORDS and len(w) > 3}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_names(chapter: dict, aliases: dict[str, str] | None = None) -> dict:
    """Unifica nombres de personaje en characters, events y relationships.

    Conserva los nombres originales en ``characters_raw`` (trazabilidad).
    No altera ``summary`` ni ``locations``.
    """
    raw = list(chapter.get("characters", []))
    if "characters_raw" not in chapter:
        chapter["characters_raw"] = raw
    chapter["characters"] = _dedupe_preserving_order(
        [canonical_name(n, aliases) for n in raw]
    )

    for ev in chapter.get("events", []):
        ev["text"] = normalize_text(ev["text"], aliases)
    for rel in chapter.get("relationships", []):
        rel["relation"] = normalize_text(rel["relation"], aliases)
    return chapter


def dedupe_events(chapter: dict) -> dict:
    """Fusiona eventos casi identicos adyacentes dentro del mismo capitulo.

    Greedy sobre la lista en orden: si dos eventos consecutivos tienen Jaccard
    >= DEDUPE_JACCARD se fusionan en uno solo (texto mas informativo, union de
    ``source_chunks``). Los eventos distintos, aunque compartan tema, no se
    tocan.
    """
    events = chapter.get("events", [])
    merged: list[dict] = []
    for ev in events:
        if merged and _jaccard(
            _content_tokens(merged[-1]["text"]), _content_tokens(ev["text"])
        ) >= DEDUPE_JACCARD:
            prev = merged[-1]
            if len(ev["text"]) > len(prev["text"]):
                prev["text"] = ev["text"]
            prev["source_chunks"] = sorted(
                set(prev.get("source_chunks", [])) | set(ev.get("source_chunks", []))
            )
        else:
            merged.append(dict(ev))
    chapter["events"] = merged
    return chapter


def postprocess_chapter(chapter: dict, aliases: dict[str, str] | None = None) -> dict:
    """Aplica normalizacion de nombres y dedupe de eventos a un capitulo."""
    normalize_names(chapter, aliases)
    dedupe_events(chapter)
    return chapter


def postprocess_memory(
    data: dict, aliases: dict[str, str] | Path | None = None
) -> dict:
    """Aplica el post-procesado a todos los capitulos y registra el informe.

    ``aliases`` puede ser un dict {alias: canonico} o una ruta a un JSON
    (ver ``load_aliases``). El informe (``meta.postprocess``) lista los alias
    realmente aplicados y el numero de eventos fusionados por capitulo, para
    auditoria y reversibilidad.
    """
    if aliases is not None and not isinstance(aliases, dict):
        aliases = load_aliases(aliases)
    aliases = aliases or {}
    aliases_applied: set[str] = set()
    chapters_report: dict[str, dict] = {}
    for chapter in data.get("chapters", []):
        before = len(chapter.get("events", []))
        normalize_names(chapter, aliases)
        for alias in aliases:
            if any(alias in n.lower() for n in chapter.get("characters_raw", [])):
                aliases_applied.add(alias)
        dedupe_events(chapter)
        after = len(chapter.get("events", []))
        chapters_report[str(chapter.get("chapter_index"))] = {
            "events_before": before,
            "events_after": after,
            "events_merged": before - after,
        }
    data.setdefault("meta", {})["postprocess"] = {
        "version": 1,
        "dedupe_jaccard": DEDUPE_JACCARD,
        "aliases_applied": sorted(aliases_applied),
        "aliases": aliases,
        "chapters": chapters_report,
    }
    return data
