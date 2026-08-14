"""Experimento controlado: baseline vs memoria narrativa vs hybrid (Fase 2C).

Compara las mismas 16 preguntas de ``data/eval_questions.json`` con tres
estrategias de retrieval y mide el impacto REAL en el sistema completo:

  - baseline       : Qdrant 700/100 (expansion=off, rerank=none). Reutiliza los
                     resultados ya guardados de ``results_baseline700.json``.
  - narrative      : solo memoria narrativa como localizador (para aislar su
                     contribucion). El texto final se resuelve en Qdrant.
  - hybrid         : fusion baseline + memoria (min-max + pesos). Dedupe de
                     candidatos y seleccion de los top-k finales (presupuesto
                     del LLM, por defecto 8, no 8+8).

La memoria narrativa SOLO localiza chunks (``source_chunks``/``chunk_refs``):
nunca aporta texto. No se usa planner, multi-query, MMR ni otro LLM.

Genera respuestas con el mismo LLM y prompt de la Fase 2A para narrative e
hybrid (baseline ya las tiene guardadas), de modo que la clasificacion manual
(grades) pueda rellenarse igual que en esa fase.

Salida:
  data/eval_answers/hybrid_<ts>.json            (detalle por pregunta, 3 modos)
  data/eval_answers/results_hybrid700_narrative.json   (formato results_*.json)
  data/eval_answers/results_hybrid700_hybrid.json
  data/eval_answers/grades_hybrid700_narrative.json    (huecos para evaluacion manual)
  data/eval_answers/grades_hybrid700_hybrid.json
  (informe comparativo: python scripts/build_hybrid_report.py)

Uso:
  python scripts/evaluate_hybrid.py --collection narrative_c700_o100
  python scripts/evaluate_hybrid.py --collection narrative_c700_o100 --retrieval-only
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings, get_settings  # noqa: E402
from app.llm.prompts import build_qa_messages  # noqa: E402
from app.memory.retrieval import build_units, embed_units, score_question  # noqa: E402
from app.models.schemas import SearchHit  # noqa: E402
from app.retrieval.hybrid import expand_narrative_chapters  # noqa: E402
from app.service import Service  # noqa: E402

# Rubrica identica a evaluate_answers.py (Fase 2A), para grades compatibles.
GRADES_RUBRIC = {
    "correcta": "Responde correctamente con evidencia suficiente del contexto.",
    "parcial": "Respuesta util pero incompleta, ambigua o con parte de la evidencia mal usada.",
    "incorrecta": "Responde algo incorrecto aunque el contexto contenía la evidencia.",
    "sin_evidencia": "El contexto recuperado no bastaba y el modelo lo declara o no puede responder.",
    "alucinacion": "Afirma hechos no respaldados por el contexto recuperado.",
}

KS = [5, 8]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "eval_answers"


def _recall(expected: set[int], got: list[int]) -> float:
    if not expected:
        return 0.0
    return len(expected & set(got)) / len(expected)


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
    timed = [r for r in rows if r.get("generation_s") is not None]
    summary["mean_retrieval_s"] = round(sum(r["retrieval_s"] for r in rows) / n, 3)
    summary["mean_chunks"] = round(sum(len(r["retrieved_chapters"]) for r in rows) / n, 2)
    if timed:
        summary["mean_generation_s"] = round(sum(r["generation_s"] for r in timed) / n, 3)
        summary["mean_total_s"] = round(sum(r["total_s"] for r in timed) / n, 3)
        with_tokens = [r for r in timed if r.get("prompt_tokens") is not None]
        if with_tokens:
            summary["mean_prompt_tokens"] = round(
                sum(r["prompt_tokens"] for r in with_tokens) / len(with_tokens), 1
            )
            summary["mean_completion_tokens"] = round(
                sum(r["completion_tokens"] for r in with_tokens) / len(with_tokens), 1
            )
    return summary


def _baseline_rows(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {r["question"]: r for r in data["rows"]}


def _row_baseline(item: dict, b: dict) -> dict:
    expected = set(item.get("expected_chapters", []))
    got = b["retrieved_chapters"]
    row = {
        "question": item["question"],
        "expected_chapters": sorted(expected),
        "expected_facts": item.get("expected_facts", []),
        "mode": "baseline",
        "retrieved_chapters": got,
        "retrieval_s": b["retrieval_s"],
        "chunks": b.get("chunks", []),
        "context": b.get("context"),
        "answer": b.get("answer"),
        "prompt_tokens": b.get("prompt_tokens"),
        "completion_tokens": b.get("completion_tokens"),
        "generation_s": b.get("generation_s"),
        "total_s": b.get("total_s"),
        "source_counts": None,
    }
    for k in KS:
        row[f"recall@{k}"] = round(_recall(expected, got[:k]), 3)
        row[f"any@{k}"] = bool(expected & set(got[:k]))
    return row


def _chunks_from_hits(hits) -> list[dict]:
    return [
        {
            "chapter_index": h.chunk.chapter_index,
            "chunk_index": h.chunk.chunk_index,
            "score": round(h.score, 4),
            "global_position": h.chunk.global_position,
            "preview": h.chunk.text[:200],
        }
        for h in hits
    ]


def _narrative_only(
    service,
    item: dict,
    units: list[dict],
    memory: dict,
    book_id: str,
    top_k: int,
    settings: Settings,
) -> dict:
    """Solo memoria narrativa: rankea capitulos y expande a chunks originales.

    Mantiene el mismo presupuesto (top_k chunks) y resuelve el texto en Qdrant.
    """
    expected = set(item.get("expected_chapters", []))
    question = item["question"]
    t0 = time.perf_counter()
    q_emb = service._embedder.embed([question])[0]
    ranked = score_question(question, q_emb, units)
    cands = expand_narrative_chapters(
        ranked,
        memory["chapters"],
        top_n=settings.hybrid_narrative_top,
        chunks_per_chapter=settings.hybrid_chunks_per_chapter,
    )
    cands.sort(key=lambda c: c["narrative_score"], reverse=True)
    cands = cands[:top_k]
    chunks = service._store.get_chunks(book_id, (c["chunk_index"] for c in cands))
    hits = [
        {
            "chapter_index": c["chapter_index"],
            "chunk_index": c["chunk_index"],
            "score": round(c["narrative_score"], 4),
            "global_position": chunks[c["chunk_index"]].global_position,
            "preview": chunks[c["chunk_index"]].text[:200],
        }
        for c in cands
        if c["chunk_index"] in chunks
    ]
    retrieval_s = round(time.perf_counter() - t0, 3)
    got = [h["chapter_index"] for h in hits]
    row = {
        "question": question,
        "expected_chapters": sorted(expected),
        "expected_facts": item.get("expected_facts", []),
        "mode": "narrative",
        "retrieved_chapters": got,
        "retrieval_s": retrieval_s,
        "chunks": hits,
        "source_counts": None,
        "narrative_ranked": [
            {"chapter_index": r["chapter_index"], "score": r["score"]}
            for r in ranked[: settings.hybrid_narrative_top]
        ],
    }
    for k in KS:
        row[f"recall@{k}"] = round(_recall(expected, got[:k]), 3)
        row[f"any@{k}"] = bool(expected & set(got[:k]))
    return row


def _hybrid(
    service,
    item: dict,
    units: list[dict],
    memory: dict,
    book_id: str,
    top_k: int,
    settings: Settings,
) -> dict:
    expected = set(item.get("expected_chapters", []))
    question = item["question"]
    t0 = time.perf_counter()
    hr = service.search_hybrid(
        question,
        units,
        memory["chapters"],
        top_k=top_k,
        book_id=book_id,
    )
    retrieval_s = round(time.perf_counter() - t0, 3)
    hits = _chunks_from_hits(hr.result.hits)
    got = [h["chapter_index"] for h in hits]
    counts = {
        "from_baseline_only": sum(
            1 for c in hr.contributions if c["from_baseline"] and not c["from_narrative"]
        ),
        "from_narrative_only": sum(
            1 for c in hr.contributions if c["from_narrative"] and not c["from_baseline"]
        ),
        "from_both": sum(
            1 for c in hr.contributions if c["from_baseline"] and c["from_narrative"]
        ),
    }
    row = {
        "question": question,
        "expected_chapters": sorted(expected),
        "expected_facts": item.get("expected_facts", []),
        "mode": "hybrid",
        "retrieved_chapters": got,
        "retrieval_s": retrieval_s,
        "chunks": hits,
        "contributions": hr.contributions,
        "source_counts": counts,
        "narrative_ranked": hr.narrative_ranked,
    }
    for k in KS:
        row[f"recall@{k}"] = round(_recall(expected, got[:k]), 3)
        row[f"any@{k}"] = bool(expected & set(got[:k]))
    return row


def _generate(service, row: dict, chunks: list[dict], question: str, book_id: str) -> dict:
    """Genera la respuesta con el mismo prompt/LLM de la Fase 2A."""
    # Necesitamos el texto completo de los chunks, no el preview. Se resuelve en
    # Qdrant a partir de los chunk_index (la memoria solo localiza).
    full = service._store.get_chunks(book_id, (c["chunk_index"] for c in chunks))

    real_hits = []
    for c in chunks:
        ch = full.get(c["chunk_index"])
        if ch is None:
            continue
        real_hits.append(SearchHit(chunk=ch, score=c["score"]))
    messages = build_qa_messages(question, real_hits)
    context_text = messages[1]["content"]
    t0 = time.perf_counter()
    chat = service._llm.chat_detailed(messages)
    generation_s = round(time.perf_counter() - t0, 3)
    row["context"] = context_text
    row["answer"] = chat.content
    row["prompt_tokens"] = chat.prompt_tokens
    row["completion_tokens"] = chat.completion_tokens
    row["generation_s"] = generation_s
    row["total_s"] = round(row["retrieval_s"] + generation_s, 3)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimento hybrid (Fase 2C)")
    parser.add_argument("--eval-file", default=str(DEFAULT_OUT.parent / "eval_questions.json"))
    parser.add_argument("--collection", default=None, help="Coleccion Qdrant (narrative_c700_o100)")
    parser.add_argument("--baseline", default=str(DEFAULT_OUT / "results_baseline700.json"))
    parser.add_argument("--memory", default=str(DEFAULT_OUT.parent / "narrative_memory.json"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--label", default="hybrid700")
    parser.add_argument("--retrieval-only", action="store_true", help="Saltar la generacion del LLM")
    parser.add_argument("--no-cache", action="store_true", help="Recomputar embeddings de la memoria")
    args = parser.parse_args()

    label = args.label
    settings = get_settings()
    if args.collection:
        settings.collection_name = args.collection
    service = Service(settings)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    memory = json.loads(Path(args.memory).read_text(encoding="utf-8"))
    units = build_units(memory)
    cache_path = out_dir / "narrative_memory_units_emb.json"
    units = embed_units(
        service._embedder, units, cache_path=cache_path, use_cache=not args.no_cache
    )

    eval_data = json.loads(Path(args.eval_file).read_text(encoding="utf-8"))
    book_id = eval_data.get("book_id")
    top_k = eval_data.get("top_k") or 8
    baseline_rows = _baseline_rows(Path(args.baseline))

    rows: dict[str, list[dict]] = {"baseline": [], "narrative": [], "hybrid": []}
    try:
        for item in eval_data["questions"]:
            q = item["question"]
            rows["baseline"].append(_row_baseline(item, baseline_rows[q]))
            narr = _narrative_only(service, item, units, memory, book_id, top_k, settings)
            if not args.retrieval_only:
                narr = _generate(service, narr, narr["chunks"], q, book_id)
            rows["narrative"].append(narr)
            hyb = _hybrid(service, item, units, memory, book_id, top_k, settings)
            if not args.retrieval_only:
                hyb = _generate(service, hyb, hyb["chunks"], q, book_id)
            rows["hybrid"].append(hyb)
            print(
                f"[hybrid] recall@8={hyb['recall@8']:.2f} (baseline {rows['baseline'][-1]['recall@8']:.2f}) "
                f"{q[:45]}"
            )
    finally:
        service.close()

    summaries = {mode: _metrics(rws) for mode, rws in rows.items()}
    result = {
        "meta": {
            "timestamp": timestamp,
            "label": label,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "collection": settings.collection_name,
            "memory": Path(args.memory).name,
            "baseline": Path(args.baseline).name,
            "top_k": top_k,
            "hybrid": {
                "narrative_top": settings.hybrid_narrative_top,
                "chunks_per_chapter": settings.hybrid_chunks_per_chapter,
                "weight_baseline": settings.hybrid_weight_baseline,
                "weight_narrative": settings.hybrid_weight_narrative,
            },
            "retrieval_only": args.retrieval_only,
        },
        "summary": summaries,
        "rows": rows,
    }

    stamp_name = f"hybrid_{timestamp}.json"
    (out_dir / stamp_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "hybrid.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # results_*.json en el formato de Fase 2A (compatibles con summarize_answers.py)
    for mode in ("narrative", "hybrid"):
        results_like = {
            "meta": {
                "timestamp": timestamp,
                "label": f"{label}_{mode}",
                "llm_model": settings.llm_model,
                "llm_think": settings.llm_think,
                "embedding_model": settings.embedding_model,
                "chunk_tokens": settings.chunk_tokens,
                "chunk_overlap": settings.chunk_overlap,
                "collection": settings.collection_name,
                "retrieval": f"{mode} (experimental Fase 2C)",
                "top_k": top_k,
                "retrieval_only": args.retrieval_only,
            },
            "summary": summaries[mode],
            "rows": rows[mode],
        }
        (out_dir / f"results_{label}_{mode}.json").write_text(
            json.dumps(results_like, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not args.retrieval_only:
            grades = {
                "rubric": GRADES_RUBRIC,
                "questions": [
                    {
                        "question": r["question"],
                        "expected_chapters": r["expected_chapters"],
                        "expected_facts": r.get("expected_facts", []),
                        "expected_answer": None,
                        "grade": None,
                        "notes": "",
                    }
                    for r in rows[mode]
                ],
            }
            (out_dir / f"grades_{label}_{mode}.json").write_text(
                json.dumps(grades, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    print("\n" + "=" * 70)
    for mode in ("baseline", "narrative", "hybrid"):
        s = summaries[mode]
        print(f"\n[{mode}]")
        for key in (
            "mean_recall@5",
            "mean_recall@8",
            "any_rate@5",
            "any_rate@8",
            "multi_recall@8",
            "retrieval_fail_any@8",
            "mean_retrieval_s",
        ):
            print(f"  {key}: {s.get(key)}")
    print(f"\nJSON:  {out_dir / stamp_name}")


if __name__ == "__main__":
    main()
