"""Consolida los experimentos de retrieval (Fase 1.5) en summary.json.

Lee data/experiments/detail_*.json y recalcula los agregados por config de
chunk y estrategia, reescribiendo data/experiments/summary.json con todas las
configuraciones (cada ejecucion parcial del harness solo guardaba la suya).
Tambien imprime la tabla comparativa y el mejor resultado por metrica.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

def _eval_book_id() -> str:
    path = ROOT / "data" / "eval_questions.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("book_id", "desconocido")
    return "desconocido"
EXPERIMENTS = ROOT / "data" / "experiments"
KS = [5, 8, 10]
STRATEGY_ORDER = ["baseline", "multi-query", "multi-query+mmr"]


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    multi = [r for r in rows if r["multi_chapter"]]
    agg: dict = {"questions": n, "multi_chapter_questions": len(multi)}
    for k in KS:
        agg[f"mean_recall@{k}"] = round(sum(r[f"recall@{k}"] for r in rows) / n, 3)
        agg[f"any_rate@{k}"] = round(sum(int(r[f"any@{k}"]) for r in rows) / n, 3)
        if multi:
            agg[f"multi_recall@{k}"] = round(sum(r[f"recall@{k}"] for r in multi) / len(multi), 3)
    agg["mean_queries"] = round(sum(r["n_queries"] for r in rows) / n, 2)
    agg["mean_elapsed_s"] = round(sum(r["elapsed_s"] for r in rows) / n, 4)
    return agg


def main() -> None:
    configs: dict = {}
    for detail in sorted(EXPERIMENTS.glob("detail_*.json")):
        collection = detail.name[len("detail_") : -len(".json")]
        rows = json.loads(detail.read_text(encoding="utf-8"))["rows"]
        by_strategy = {}
        for r in rows:
            by_strategy[r["strategy"]] = r
        aggregates = {s: aggregate([r for r in rows if r["strategy"] == s]) for s in STRATEGY_ORDER}
        # metadata de ingest: la reconstruimos a partir del nombre de la coleccion
        tokens, overlap = _parse_collection(collection)
        configs[collection] = {
            "chunk_tokens": tokens,
            "chunk_overlap": overlap,
            "chunk_count": _chunk_count(collection),
            "aggregates": aggregates,
        }

    summary = {
        "book_id": _eval_book_id(),
        "questions": 16,
        "ks": KS,
        "strategies": {
            "baseline": {"expansion": "off", "rerank": "none"},
            "multi-query": {"expansion": "heuristic", "rerank": "none"},
            "multi-query+mmr": {"expansion": "heuristic", "rerank": "mmr"},
        },
        "configs": configs,
    }
    (EXPERIMENTS / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{'config':<12} {'chunks':>6} | " + " | ".join(f"baseline recall@{k:>2}" for k in KS)
          + " | " + " | ".join(f"mq+mmr recall@{k:>2}" for k in KS) + " | t/query(s)")
    print("-" * 90)
    best = {}
    for collection, cfg in configs.items():
        b = cfg["aggregates"]["baseline"]
        m = cfg["aggregates"]["multi-query+mmr"]
        label = f"{cfg['chunk_tokens']}/{cfg['chunk_overlap']}"
        print(f"{label:<12} {cfg['chunk_count']:>6} | "
              + " | ".join(f"{b[f'mean_recall@{k}']:.3f}" for k in KS) + " | "
              + " | ".join(f"{m[f'mean_recall@{k}']:.3f}" for k in KS)
              + f" | {b['mean_elapsed_s']:.2f}")
        for k in KS:
            best[(k, "baseline")] = max(best.get((k, "baseline"), -1), b[f"mean_recall@{k}"])
            best[(k, "mq+mmr")] = max(best.get((k, "mq+mmr"), -1), m[f"mean_recall@{k}"])
    print(f"\nsummary.json actualizado: {EXPERIMENTS / 'summary.json'}")


def _parse_collection(name: str) -> tuple[int, int]:
    if name == "narrative_chunks":
        return 500, 50
    parts = name.split("_")  # narrative_c300_o50 -> c300, o50
    tokens = int(parts[1][1:])
    overlap = int(parts[2][1:])
    return tokens, overlap


def _chunk_count(collection: str) -> int | None:
    if collection == "narrative_chunks":
        return 448
    logs = [
        EXPERIMENTS.parent / "experiments_run.log",
        EXPERIMENTS.parent / "experiments_run_700_50.log",
    ]
    for log in logs:
        if not log.exists():
            continue
        try:
            text = log.read_text(encoding="utf-16")
        except (UnicodeDecodeError, UnicodeError):
            text = log.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if f"-> {collection}" in line and "chunks" in line:
                try:
                    return int(line.split("'chunks': ")[1].split(",")[0])
                except (IndexError, ValueError):
                    return None
    return None


if __name__ == "__main__":
    main()
