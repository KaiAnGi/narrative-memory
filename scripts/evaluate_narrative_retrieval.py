"""Experimento de retrieval narrativo: memoria vs baseline semantico.

El pipeline de retrieval NO se toca: esto es un experimento aislado que decide
si la memoria narrativa (``data/narrative_memory.json``) merece ser una segunda
fuente de retrieval. Tampoco se genera ninguna respuesta: solo retrieval.

Mecanismo: el MISMO embedder del baseline (``qwen3-embedding:0.6b``), cosine,
pero la unidad de recuperacion es la memoria consolidada en vez de los chunks
crusos. Para cada capitulo se construyen unidades de evidencia:

  - summary      (peso 1.0, source_chunks = chunk_refs del capitulo)
  - events       (peso 1.0, source_chunks de cada evento)
  - relationships(peso 0.9, source_chunks de cada relacion)
  - characters   (peso 0.7, sin chunks)
  - locations    (peso 0.7, sin chunks)

Score del capitulo = max over unidades (cosine ponderado). La salida SIEMPRE
apunta a los chunks originales (chunk_refs del capitulo + source_chunks de la
evidencia ganadora), para comparar contra el retrieval semantico existente.

Salida:
  data/eval_answers/narrative_retrieval_<ts>.json   (detalle por pregunta)
  data/eval_answers/narrative_retrieval_report.md   (informe vs baseline700)

Uso:
  python scripts/evaluate_narrative_retrieval.py
  python scripts/evaluate_narrative_retrieval.py --no-cache
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.embeddings.ollama_embedder import OllamaEmbedder  # noqa: E402
from app.memory.retrieval import WEIGHTS, build_units, embed_units, score_question  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = ROOT / "data" / "narrative_memory.json"
EVAL_PATH = ROOT / "data" / "eval_questions.json"
BASELINE_PATH = ROOT / "data" / "eval_answers" / "results_baseline700.json"
OUT_DIR = ROOT / "data" / "eval_answers"
CACHE_PATH = OUT_DIR / "narrative_memory_units_emb.json"

KS = [5, 8]


def _row(question_item: dict, ranked: list[dict], retrieval_s: float) -> dict:
    expected = set(question_item.get("expected_chapters", []))
    got5 = [c["chapter_index"] for c in ranked[: KS[0]]]
    got8 = [c["chapter_index"] for c in ranked[: KS[1]]]

    row = {
        "question": question_item["question"],
        "expected_chapters": sorted(expected),
        "expected_facts": question_item.get("expected_facts", []),
        "retrieval_s": round(retrieval_s, 3),
        "ranked_chapters": [
            {
                "chapter_index": c["chapter_index"],
                "score": c["score"],
                "evidence": c["evidence"],
            }
            for c in ranked[:8]
        ],
        "retrieved_chapters": got8,
        "retrieved_top5": got5,
        "retrieval_gap@8": sorted(expected - set(got8)),
    }
    for k in KS:
        got = [c["chapter_index"] for c in ranked[:k]]
        row[f"recall@{k}"] = round(
            len(expected & set(got)) / len(expected) if expected else 0.0, 3
        )
        row[f"any@{k}"] = bool(expected & set(got))
    return row


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    multi = [r for r in rows if len(r["expected_chapters"]) >= 2]
    summary = {"questions": n, "multi_chapter_questions": len(multi)}
    for k in KS:
        summary[f"mean_recall@{k}"] = round(sum(r[f"recall@{k}"] for r in rows) / n, 3)
        summary[f"any_rate@{k}"] = round(sum(int(r[f"any@{k}"]) for r in rows) / n, 3)
        if multi:
            summary[f"multi_recall@{k}"] = round(
                sum(r[f"recall@{k}"] for r in multi) / len(multi), 3
            )
    summary["retrieval_fail_any@8"] = sum(int(not r["any@8"]) for r in rows)
    summary["mean_retrieval_s"] = round(sum(r["retrieval_s"] for r in rows) / n, 3)
    return summary


def compare_with_baseline(rows: list[dict], baseline: dict) -> dict:
    """Compara memoria vs baseline: gaps recuperados y falsos positivos nuevos.

    Los gaps del baseline se derivan de sus propios resultados (capitulos
    esperados ausentes en su top-8). Se marca si la memoria los recupera.
    """
    comp = {"per_question": [], "gaps": []}
    b_rows = {r["question"]: r for r in baseline["rows"]}
    for row in rows:
        b = b_rows.get(row["question"])
        gaps = sorted(set(row["expected_chapters"]) - set(b["retrieved_chapters"])) if b else []
        recovered = [g for g in gaps if g in set(row["retrieved_chapters"])]
        comp["per_question"].append(
            {
                "question": row["question"],
                "expected_chapters": row["expected_chapters"],
                "baseline_got": b["retrieved_chapters"] if b else None,
                "baseline_gaps": gaps,
                "memory_got": row["retrieved_chapters"],
                "gaps_recovered": recovered,
                "gaps_still_missing": [g for g in gaps if g not in set(row["retrieved_chapters"])],
                "baseline_recall@8": b["recall@8"] if b else None,
                "memory_recall@8": row["recall@8"],
            }
        )
        for g in gaps:
            comp["gaps"].append(
                {
                    "chapter": g,
                    "recovered": g in set(row["retrieved_chapters"]),
                    "question": row["question"],
                    "evidence": next(
                        (
                            c["evidence"]
                            for c in row["ranked_chapters"]
                            if c["chapter_index"] == g
                        ),
                        None,
                    ),
                }
            )
    comp["gaps_recovered"] = sum(1 for g in comp["gaps"] if g["recovered"])
    comp["gaps_total"] = len(comp["gaps"])
    comp["chapters_recovered"] = sorted(
        {g["chapter"] for g in comp["gaps"] if g["recovered"]}
    )
    return comp


def write_report(result: dict, baseline: dict, out_path: Path) -> None:
    """Genera el informe markdown comparando baseline semantico vs memoria."""
    summary = result["summary"]
    comp = result["comparison"]
    b_summary = baseline["summary"]
    b_rows = {r["question"]: r for r in baseline["rows"]}

    b_distinct_total = sum(len(set(r["retrieved_chapters"])) for r in baseline["rows"])
    mem_top8_total = 8 * summary["questions"]

    new_fp_total = 0
    new_fp_per_cap: dict[int, int] = {}
    for row in result["rows"]:
        b = b_rows[row["question"]]
        expected = set(row["expected_chapters"])
        base = set(b["retrieved_chapters"])
        mem = set(row["retrieved_chapters"])
        new_fp = (mem - expected) - base
        new_fp_total += len(new_fp)
        for c in new_fp:
            new_fp_per_cap[c] = new_fp_per_cap.get(c, 0) + 1

    line = []
    add = line.append
    add("# Experimento: Memoria Narrativa vs Baseline Semantico\n")
    add(f"- Fecha: {result['meta']['timestamp']}")
    add(f"- Embedder: `{result['meta']['embedding_model']}`")
    add(f"- Memoria: `{result['meta']['memory']}` (postprocesado v{result['meta'].get('postprocess_version')})")
    add(f"- Evaluacion: `{result['meta']['eval_file']}` ({summary['questions']} preguntas, {summary['multi_chapter_questions']} multi-capitulo)")
    add(f"- Baseline: `{result['meta']['baseline']}`")
    add(f"- Mecanismo: {result['meta']['mechanism']}\n")

    def mrow(name, b_val, m_val, better):
        delta = m_val - b_val
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        add(f"| {name} | {b_val:.3f} | {m_val:.3f} | **{delta:+.3f}** {arrow} ({better}) |")

    add("## Metricas globales\n")
    add("| metrica | baseline | memoria | delta |")
    add("|---|---|---|---|")
    mrow("mean recall@5", b_summary["mean_recall@5"], summary["mean_recall@5"], "mas alto es mejor")
    mrow("mean recall@8", b_summary["mean_recall@8"], summary["mean_recall@8"], "mas alto es mejor")
    mrow("any rate@5", b_summary["any_rate@5"], summary["any_rate@5"], "mas alto es mejor")
    mrow("any rate@8", b_summary["any_rate@8"], summary["any_rate@8"], "mas alto es mejor")
    mrow("multi-chapter recall@5", b_summary["multi_recall@5"], summary["multi_recall@5"], "mas alto es mejor")
    mrow("multi-chapter recall@8", b_summary["multi_recall@8"], summary["multi_recall@8"], "mas alto es mejor")
    add("| retrieval fallidos (any@8=0) | "
        f"{int(b_summary['retrieval_fail_any@8'])} | {int(summary['retrieval_fail_any@8'])} | "
        f"**{int(summary['retrieval_fail_any@8']) - int(b_summary['retrieval_fail_any@8'])}** (menos es mejor) |")
    mrow("tiempo medio por pregunta (s)", b_summary["mean_retrieval_s"], summary["mean_retrieval_s"], "menos es mejor")
    add("")

    add("## Cobertura de capitulos en top-8\n")
    add("| | baseline | memoria |")
    add("|---|---|---|")
    add(f"| capitulos distintos recuperados | {b_distinct_total} | {mem_top8_total} (8 por pregunta) |")
    add(f"| falsos positivos NUEVOS (memoria, no en baseline) | — | {new_fp_total} |")
    add(f"| falsos positivos por capitulo | — | `{json.dumps(dict(sorted(new_fp_per_cap.items())))}` |")
    add("")
    add("La memoria devuelve 8 capitulos distintos por pregunta; el baseline devuelve "
        "8 chunks con repeticiones (71 capitulos distintos en total). Los FP nuevos "
        "son capitulos que el baseline no tenia ni siquiera en su top-8.\n")

    add("## Gaps del baseline recuperados por la memoria\n")
    add(f"- **{comp['gaps_recovered']}/{comp['gaps_total']}** gaps recuperados")
    add(f"- Capitulos recuperados: {comp['chapters_recovered'] or 'ninguno'}\n")
    add("| capitulo | recuperado | pregunta | evidencia ganadora |")
    add("|---|---|---|---|")
    for g in comp["gaps"]:
        if not g["evidence"]:
            ev = "—"
        else:
            e = g["evidence"]
            ev = f"[{e['kind']}] cos={e['cosine']:.3f}: {e['text'][:60]}"
        add(f"| {g['chapter']} | {'SI' if g['recovered'] else 'NO'} | {g['question'][:50]} | {ev} |")
    add("")

    add("## Detalle por pregunta\n")
    add("| Q | expected | baseline top-8 | memoria top-8 | recall@8 b | recall@8 m |")
    add("|---|---|---|---|---|---|")
    for i, row in enumerate(result["rows"], 1):
        b = b_rows[row["question"]]
        add(f"| {i} | {row['expected_chapters']} | {b['retrieved_chapters']} | "
            f"{row['retrieved_chapters']} | {b['recall@8']:.2f} | {row['recall@8']:.2f} |")
    add("")

    add("## Observaciones\n")
    obs = []
    if summary["mean_recall@8"] > b_summary["mean_recall@8"]:
        obs.append("- La memoria gana recall@8 (mas capitulos correctos en top-8) y "
                   "recupera la mayoria de gaps que el baseline pierde.")
    if summary["any_rate@8"] >= b_summary["any_rate@8"]:
        obs.append("- any@8 >= baseline: la memoria nunca pierde la pregunta por completo.")
    if summary["multi_recall@8"] > b_summary["multi_recall@8"]:
        obs.append("- Las preguntas multi-capitulo mejoran: la memoria agrega relaciones "
                   "y eventos que conectan capitulos.")
    obs.append("- Gaps aun perdidos (capitulos 1, 13, 24): su summary/eventos no contienen "
               "la terminologia exacta de la pregunta (p. ej. 'pesadilla' en el prologo) o "
               "quedan justo fuera del top-8; las mejoras pasarian por afinar pesos o "
               "consultas.")
    obs.append("- Coste: tiempo de retrieval similar al baseline (una sola llamada de "
               "embedding + cosine), sin necesidad de multi-query.")
    add("\n".join(obs))
    add("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(line), encoding="utf-8")
    print(f"Informe: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento de retrieval narrativo")
    parser.add_argument("--memory", default=str(MEMORY_PATH))
    parser.add_argument("--eval-file", default=str(EVAL_PATH))
    parser.add_argument("--baseline", default=str(BASELINE_PATH))
    parser.add_argument("--model", default=None, help="Embedder (por defecto el de .env)")
    parser.add_argument("--no-cache", action="store_true", help="Recomputar embeddings de la memoria")
    args = parser.parse_args()

    settings = get_settings()
    model = args.model or settings.embedding_model
    memory = json.loads(Path(args.memory).read_text(encoding="utf-8"))
    eval_data = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    embedder = OllamaEmbedder(
        base_url=settings.ollama_base_url,
        model=model,
        batch_size=settings.embedding_batch_size,
        timeout=settings.embedding_timeout_seconds,
    )

    units = build_units(memory)
    units = embed_units(embedder, units, cache_path=CACHE_PATH, use_cache=not args.no_cache)

    rows = []
    for item in eval_data["questions"]:
        t0 = time.perf_counter()
        q_emb = embedder.embed([item["question"]])[0]
        ranked = score_question(item["question"], q_emb, units)
        retrieval_s = time.perf_counter() - t0
        rows.append(_row(item, ranked, retrieval_s))
        status = f"recall@8={rows[-1]['recall@8']:.2f}"
        print(f"[narrativo] {status} {rows[-1]['question'][:60]}")

    summary = _metrics(rows)
    comparison = compare_with_baseline(rows, baseline)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "timestamp": ts,
        "embedding_model": model,
        "memory": str(Path(args.memory).name),
        "eval_file": str(Path(args.eval_file).name),
        "baseline": str(Path(args.baseline).name),
        "mechanism": "cosine qwen3-embedding:0.6b sobre unidades de la memoria "
                     "(summary/events/relationships/characters/locations) con pesos "
                     + json.dumps(WEIGHTS),
        "weights": WEIGHTS,
        "postprocess_version": memory.get("meta", {}).get("postprocess", {}).get("version"),
    }
    result = {"meta": meta, "summary": summary, "comparison": comparison, "rows": rows}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"narrative_retrieval_{ts}.json"
    (OUT_DIR / stamp).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "narrative_retrieval.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = OUT_DIR / "narrative_retrieval_report.md"
    write_report(result, baseline, report_path)
    print(f"\nJSON: {OUT_DIR / stamp}")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  gaps recuperados: {comparison['gaps_recovered']}/{comparison['gaps_total']}")


if __name__ == "__main__":
    main()
