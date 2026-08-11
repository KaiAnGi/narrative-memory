"""Experimentos de retrieval (Fase 1.5): estrategias x tamanos de chunk.

Compara estrategias de recuperacion y configuraciones de chunking usando LAS
MISMAS preguntas de data/eval_questions.json. El codigo es generico: no
contiene reglas sobre capitulos ni libros concretos.

Estrategias (RetrievalOptions):
  - baseline        : expansion off + rerank none   (= V1)
  - multi-query     : expansion heuristica + rerank none
  - multi-query+mmr : expansion heuristica + rerank MMR

Tamanos de chunk (tokens/overlap): 300/50, 500/50 y 700/100. Cada config se
indexa en una coleccion propia (narrative_c300_o50, ...). La coleccion V1
(narrative_chunks, 500/50) puede reutilizarse con --reuse-collection.

Metricas por pregunta, para k=5, 8 y 10:
  - recall@k = esperados encontrados / esperados
  - acierto con al menos un capitulo esperado
Agregados: recall medio, tasa de acierto y recall medio de preguntas
multi-capitulo (>=2 capitulos esperados). Tambien: tiempo medio por consulta
y numero medio de consultas embebidas.

Los resultados del baseline (data/eval_questions.results.json) no se tocan;
toda salida nueva va a --out (por defecto data/experiments/).

Uso:
  python scripts/evaluate_retrieval_experiments.py --book data/books/NOVELA.docx
      [--eval-file data/eval_questions.json]
      [--out data/experiments]
      [--skip-ingest] [--reingest]
      [--reuse-collection "TOK:OVER:COLECCION"]   # repite la flag por config
      [--time-llm N]                              # cronometra N expansiones LLM reales
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.retrieval.options import RetrievalOptions  # noqa: E402
from app.service import Service  # noqa: E402

MAX_K = 10
KS = [5, 8, 10]

STRATEGIES = {
    "baseline": RetrievalOptions(expansion="off", rerank="none", candidates_per_query=12),
    "multi-query": RetrievalOptions(
        expansion="heuristic", rerank="none", candidates_per_query=12, max_queries=4
    ),
    "multi-query+mmr": RetrievalOptions(
        expansion="heuristic", rerank="mmr", candidates_per_query=12,
        max_queries=4, diversity_lambda=0.7, chapter_penalty=0.5,
    ),
}

DEFAULT_CONFIGS = [
    ("300", "50", "narrative_c300_o50"),
    ("500", "50", "narrative_c500_o50"),
    ("700", "100", "narrative_c700_o100"),
]


@dataclass
class ChunkConfig:
    tokens: int
    overlap: int
    collection: str
    reuse: bool = False


def _build_configs(args) -> list[ChunkConfig]:
    overrides = {}
    for spec in args.reuse_collection or []:
        parts = spec.split(":")
        if len(parts) != 3:
            raise SystemExit(f"--reuse-collection espera 'TOK:OVER:COLECCION', recibio: {spec}")
        overrides[(int(parts[0]), int(parts[1]))] = parts[2]

    explicit = []
    for spec in args.chunk_configs or []:
        parts = spec.split(":")
        if len(parts) == 2:
            tokens, overlap = int(parts[0]), int(parts[1])
            collection = overrides.get((tokens, overlap), f"narrative_c{tokens}_o{overlap}")
        elif len(parts) == 3:
            tokens, overlap, collection = int(parts[0]), int(parts[1]), parts[2]
        else:
            raise SystemExit(f"--chunk-configs espera 'TOK:OVER[:COLECCION]', recibio: {spec}")
        explicit.append(ChunkConfig(tokens, overlap, collection, reuse=collection in overrides.values()))

    configs = []
    for tokens, overlap, collection in DEFAULT_CONFIGS:
        if args.chunk_configs:
            continue
        collection = overrides.get((int(tokens), int(overlap)), collection)
        configs.append(ChunkConfig(int(tokens), int(overlap), collection, reuse=collection in overrides.values()))
    return explicit or configs


def _recall(expected: set[int], got: list[int]) -> float:
    if not expected:
        return 0.0
    return len(expected & set(got)) / len(expected)


def _evaluate(service: Service, questions: list[dict], book_id: str) -> dict:
    rows = []
    for item in questions:
        question = item["question"]
        expected = set(item.get("expected_chapters", []))
        for strategy, options in STRATEGIES.items():
            t0 = time.perf_counter()
            result = service.search(question, top_k=MAX_K, book_id=book_id, options=options)
            elapsed = time.perf_counter() - t0
            got = [h.chunk.chapter_index for h in result.hits]
            row = {
                "question": question,
                "expected_chapters": sorted(expected),
                "retrieved_chapters": got,
                "n_queries": len(result.queries),
                "elapsed_s": round(elapsed, 4),
                "multi_chapter": len(expected) >= 2,
            }
            for k in KS:
                top = got[:k]
                row[f"recall@{k}"] = round(_recall(expected, top), 3)
                row[f"any@{k}"] = bool(expected & set(top))
            rows.append({"strategy": strategy, **row})
    return rows


def _aggregate(rows: list[dict]) -> dict:
    agg = {}
    for strategy in STRATEGIES:
        strat_rows = [r for r in rows if r["strategy"] == strategy]
        n = len(strat_rows)
        multi = [r for r in strat_rows if r["multi_chapter"]]
        entry = {"questions": n, "multi_chapter_questions": len(multi)}
        for k in KS:
            entry[f"mean_recall@{k}"] = round(sum(r[f"recall@{k}"] for r in strat_rows) / n, 3)
            entry[f"any_rate@{k}"] = round(sum(int(r[f"any@{k}"]) for r in strat_rows) / n, 3)
            if multi:
                entry[f"multi_recall@{k}"] = round(
                    sum(r[f"recall@{k}"] for r in multi) / len(multi), 3
                )
        entry["mean_queries"] = round(sum(r["n_queries"] for r in strat_rows) / n, 2)
        entry["mean_elapsed_s"] = round(sum(r["elapsed_s"] for r in strat_rows) / n, 4)
        agg[strategy] = entry
    return agg


def _maybe_ingest(service: Service, book: Path, cfg: ChunkConfig, args, book_id: str) -> dict:
    if cfg.reuse or args.skip_ingest:
        return {"reused_collection": cfg.collection}
    if not args.reingest:
        try:
            if service._store.count(book_id) > 0:
                return {"reused_collection": cfg.collection}
        except Exception:  # noqa: BLE001  (coleccion no existe aun)
            pass
    t0 = time.perf_counter()
    report = service.ingest_book(book)
    ingest_s = time.perf_counter() - t0
    return {
        "book_id": report.book_id,
        "paragraphs": report.paragraphs,
        "chapters": report.chapters,
        "chunks": report.chunks,
        "ingest_s": round(ingest_s, 2),
        "collection_count": service._store.count(report.book_id),
    }


def _print_table(config_label: str, agg: dict) -> None:
    print(f"\n=== {config_label} ===")
    header = f"{'estrategia':<14} | " + " | ".join(f"recall@{k:>2}" for k in KS) + " | " + \
             " | ".join(f"any@{k:>2}" for k in KS) + " | multich@8 | qry | t/query(s)"
    print(header)
    print("-" * len(header))
    for strategy, entry in agg.items():
        cells = [f"{entry[f'mean_recall@{k}']:.3f}" for k in KS]
        anys = [f"{entry[f'any_rate@{k}']:.3f}" for k in KS]
        multi = entry.get("multi_recall@8", "  -  ")
        print(
            f"{strategy:<14} | " + " | ".join(cells) + " | " + " | ".join(anys)
            + f" | {multi:>8} | {entry['mean_queries']:>3} | {entry['mean_elapsed_s']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos de retrieval (Fase 1.5)")
    parser.add_argument("--book", required=True, help="Ruta al .docx de la novela")
    parser.add_argument("--eval-file", default=str(Path(__file__).resolve().parents[1] / "data" / "eval_questions.json"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "experiments"))
    parser.add_argument("--skip-ingest", action="store_true", help="No ingerir (colecciones ya listas)")
    parser.add_argument("--reingest", action="store_true", help="Re-ingerir aunque exista el libro")
    parser.add_argument("--reuse-collection", action="append", help="Reutilizar coleccion: 'TOK:OVER:COLECCION'")
    parser.add_argument("--chunk-configs", action="append", help="Solo estas configs: 'TOK:OVER[:COLECCION]'")
    parser.add_argument("--time-llm", type=int, default=0, help="Cronometrar N expansiones con LLM real")
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    questions = data["questions"]
    book_id = data.get("book_id")
    book = Path(args.book)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = _build_configs(args)
    if not configs:
        raise SystemExit("No hay configuraciones de chunk que evaluar.")

    print(f"Preguntas: {len(questions)}  |  libro: {book}  |  book_id: {book_id}")
    combined: dict = {"book_id": book_id, "eval_file": str(eval_file), "ks": KS, "configs": {}}

    for cfg in configs:
        settings = get_settings().model_copy(
            update={
                "chunk_tokens": cfg.tokens,
                "chunk_overlap": cfg.overlap,
                "collection_name": cfg.collection,
            }
        )
        label = f"{cfg.tokens}/{cfg.overlap} -> {cfg.collection}"
        service = Service(settings)
        try:
            ingest_info = _maybe_ingest(service, book, cfg, args, book_id)
            print(f"[{label}] {ingest_info}")
            rows = _evaluate(service, questions, book_id)
            agg = _aggregate(rows)
            _print_table(label, agg)
        finally:
            service.close()

        combined["configs"][cfg.collection] = {
            "chunk_tokens": cfg.tokens,
            "chunk_overlap": cfg.overlap,
            "ingest": ingest_info,
            "aggregates": agg,
        }
        detail = out_dir / f"detail_{cfg.collection}.json"
        detail.write_text(
            json.dumps({"config": cfg.collection, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    combined["strategies"] = STRATEGIES_AS_TEXT()

    summary_file = out_dir / "summary.json"
    summary_file.write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nResultados guardados en: {out_dir}")

    if args.time_llm > 0:
        _time_llm_expansion(args.time_llm, questions[: args.time_llm])


def STRATEGIES_AS_TEXT() -> dict:
    return {
        name: {"expansion": opts.expansion, "rerank": opts.rerank}
        for name, opts in STRATEGIES.items()
    }


def _time_llm_expansion(n: int, sample_questions: list[dict]) -> None:
    from app.retrieval.query_expander import build_expander

    settings = get_settings()
    service = Service(settings)
    try:
        expander = build_expander("llm", max_queries=4, llm=service._llm)
        times = []
        for item in sample_questions:
            t0 = time.perf_counter()
            queries = expander.expand(item["question"])
            times.append(time.perf_counter() - t0)
            print(f"  LLM: {times[-1]:.2f}s -> {len(queries)} consultas | {queries[1][:60] if len(queries) > 1 else ''}")
        mean_s = sum(times) / len(times)
        print(f"\nExpansion LLM (qwen3:1.7b, think=false): media {mean_s:.2f}s por consulta")
    finally:
        service.close()


if __name__ == "__main__":
    main()
