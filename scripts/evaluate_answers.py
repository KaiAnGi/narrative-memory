"""Evaluacion del pipeline completo (Fase 2A).

pregunta -> retrieval (baseline V1) -> contexto -> qwen3:1.7b -> respuesta final

Objetivo de la fase: medir si el LLM es capaz de responder correctamente CUANDO
recibe el contexto recuperado, antes de construir agentes, memoria o knowledge
graph. No se optimiza el retrieval aqui: se mide el sistema actual.

El retrieval se ejecuta en modo baseline (expansion=off, rerank=none), la
configuracion ganadora de la Fase 1.5 con chunks 700/100 (coleccion
narrative_c700_o100; pasala con --collection si no es la de .env).

Para cada pregunta conserva: pregunta, capitulos esperados, expected_facts,
chunks recuperados (orden + score + preview), contexto COMPLETO enviado al LLM,
respuesta generada y tiempos (retrieval / generacion / total), ademas de los
tokens de prompt/completion cuando Ollama los proporciona.

La clasificacion de la respuesta (correcta / parcial / incorrecta /
sin_evidencia / alucinacion) es MANUAL: el script genera grades_<label>.json
con un campo grade por pregunta y huecos para anadir manualmente
expected_facts y expected_answer. Rellena ese archivo y despues ejecuta:

  python scripts/summarize_answers.py \
      --results data/eval_answers/results_<label>.json \
      --grades   data/eval_answers/grades_<label>.json

Modo experimento controlado de generacion (--from-results): reutiliza los
chunks/contexto EXACTOS de una ejecucion anterior y solo vuelve a generar la
respuesta con otro LLM. Asi el retrieval queda intacto y cambia unicamente el
modelo de generacion (para comparar p. ej. qwen3:1.7b vs qwen2.5:3b).

Uso:
  python scripts/evaluate_answers.py --collection narrative_c700_o100
  python scripts/evaluate_answers.py --collection narrative_c700_o100 --retrieval-only
  python scripts/evaluate_answers.py --label prueba --top-k 8
  # experimento controlado: mismo contexto, otro LLM
  python scripts/evaluate_answers.py --from-results data/eval_answers/results_baseline700.json --model qwen2.5:3b --label qwen25_3b
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
from app.llm.ollama_llm import OllamaLLM  # noqa: E402
from app.llm.prompts import SYSTEM_PROMPT, build_qa_messages  # noqa: E402
from app.retrieval.options import RetrievalOptions  # noqa: E402
from app.service import Service  # noqa: E402

KS = [5, 8, 10]
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "eval_answers"
BASELINE_OPTIONS = RetrievalOptions(expansion="off", rerank="none")

GRADES_RUBRIC = {
    "correcta": "Responde correctamente con evidencia suficiente del contexto.",
    "parcial": "Respuesta util pero incompleta, ambigua o con parte de la evidencia mal usada.",
    "incorrecta": "Responde algo incorrecto aunque el contexto contenía la evidencia.",
    "sin_evidencia": "El contexto recuperado no bastaba y el modelo lo declara o no puede responder.",
    "alucinacion": "Afirma hechos no respaldados por el contexto recuperado.",
}


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
    summary["retrieval_fail_any@8"] = sum(
        int(not r["any@8"]) for r in rows
    )
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


def _row(service, item, top_k, book_id, options, with_llm):
    question = item["question"]
    expected = set(item.get("expected_chapters", []))
    expected_facts = item.get("expected_facts", [])

    t0 = time.perf_counter()
    result = service.search(question, top_k=top_k, book_id=book_id, options=options)
    retrieval_s = round(time.perf_counter() - t0, 3)

    messages = build_qa_messages(question, result.hits)
    context_text = messages[1]["content"]
    hits = [
        {
            "chapter_index": h.chunk.chapter_index,
            "chunk_index": h.chunk.chunk_index,
            "score": round(h.score, 4),
            "global_position": h.chunk.global_position,
            "preview": h.chunk.text[:200],
        }
        for h in result.hits
    ]
    got = [h["chapter_index"] for h in hits]
    gap = sorted(expected - set(got))

    row = {
        "question": question,
        "expected_chapters": sorted(expected),
        "expected_facts": expected_facts,
        "queries": list(result.queries),
        "retrieved_chapters": got,
        "n_chunks": len(hits),
        "chunks": hits,
        "context": context_text,
        "retrieval_s": retrieval_s,
        "retrieval_ok@8": bool(expected & set(got)),
        "retrieval_gap@8": gap,
    }
    for k in KS:
        top = got[:k]
        row[f"recall@{k}"] = round(_recall(expected, top), 3)
        row[f"any@{k}"] = bool(expected & set(top))

    if with_llm:
        try:
            t0 = time.perf_counter()
            chat = service._llm.chat_detailed(messages)
            generation_s = round(time.perf_counter() - t0, 3)
            row["answer"] = chat.content
            row["prompt_tokens"] = chat.prompt_tokens
            row["completion_tokens"] = chat.completion_tokens
            row["generation_s"] = generation_s
            row["total_s"] = round(retrieval_s + generation_s, 3)
        except Exception as exc:  # noqa: BLE001
            row["answer"] = None
            row["error"] = str(exc)
    return row


def _write_report(report_path, meta, rows, summary):
    lines = [
        "# Evaluacion de respuestas (Fase 2A)",
        "",
        f"Generado: {meta['timestamp']}  |  modelo={meta['llm_model']} think={meta['llm_think']}",
        f"embeddings={meta['embedding_model']}  |  chunk={meta['chunk_tokens']}/{meta['chunk_overlap']}",
        f"top_k={meta['top_k']}  |  coleccion={meta['collection']}  |  retrieval: {meta['retrieval']}",
        "",
        "## Resumen",
        "",
        "| Metrica | Valor |",
        "|---|---|",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.3f} |")
        else:
            lines.append(f"| {key} | {value} |")
    lines += ["", "## Rubrica de clasificacion manual", ""]
    for grade, desc in GRADES_RUBRIC.items():
        lines.append(f"- **{grade}**: {desc}")
    lines += [
        "",
        "Rellena `grades_<label>.json` (campo `grade` por pregunta) y ejecuta "
        "`summarize_answers.py` para agregar la calidad de las respuestas.",
        "",
    ]

    for i, row in enumerate(rows, start=1):
        lines.append(f"## {i}. {row['question']}")
        lines.append("")
        expected = ",".join(str(c) for c in row["expected_chapters"])
        facts = row.get("expected_facts") or []
        facts_text = (
            "; ".join(f.get("fact", "") for f in facts) if facts else "(sin expected_facts aun)"
        )
        lines.append(f"- Esperados: [{expected}]  |  Recuperados (top-{len(row['retrieved_chapters'])}): {row['retrieved_chapters']}")
        lines.append(f"- Expected_facts: {facts_text}")
        if row.get("queries"):
            lines.append(f"- Consultas de retrieval: {row['queries']}")
        if row.get("retrieval_gap@8"):
            lines.append(f"- **Capítulos esperados NO recuperados en top-8: {row['retrieval_gap@8']}**")
        timing = f"retrieval {row['retrieval_s']}s"
        if row.get("generation_s") is not None:
            timing += f" · generacion {row['generation_s']}s · total {row['total_s']}s"
            tokens = "tokens no disponibles"
            if row.get("prompt_tokens") is not None:
                tokens = f"prompt {row['prompt_tokens']} / completion {row['completion_tokens']}"
            timing += f" · {tokens}"
        lines.append(f"- Tiempo: {timing}")
        lines.append("")
        lines.append("**Chunks recuperados:**")
        lines.append("")
        for h in row["chunks"]:
            lines.append(f"- cap. {h['chapter_index']} · chunk {h['chunk_index']} · score {h['score']}: {h['preview'].replace(chr(10), ' ')}...")
        lines.append("")
        lines.append("<details>")
        lines.append("<summary><b>Contexto completo enviado al LLM</b></summary>")
        lines.append("")
        lines.append(row["context"])
        lines.append("")
        lines.append("</details>")
        lines.append("")
        if row.get("answer") is not None:
            lines.append("**Respuesta generada:**")
            lines.append("")
            lines.append(f"> {row['answer']}")
            lines.append("")
        elif row.get("error"):
            lines.append(f"**Error al generar:** `{row['error']}`")
            lines.append("")
        lines.append("**Evaluacion manual:**  [ ] correcta  [ ] parcial  [ ] incorrecta  [ ] sin_evidencia  [ ] alucinacion")
        lines.append("")
        lines.append("Notas: ")
        lines.append("")
        lines.append("---")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _build_service(settings: Settings, model: str | None) -> Service:
    """Service con override opcional del modelo LLM (sin tocar .env ni el cache)."""
    llm = None
    if model:
        llm = OllamaLLM(
            base_url=settings.ollama_base_url,
            model=model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            think=settings.llm_think,
        )
    return Service(settings, llm=llm)


def _generate_rows_from_results(service: Service, src_rows: list[dict]) -> list[dict]:
    """Regenera SOLO la respuesta sobre el contexto ya guardado de un results previo.

    Mantiene intactos chunking, retrieval, scores, recall y contexto; cambia
    unicamente el LLM de generacion. El mensaje user es el texto exacto que se
    envio antes (campo ``context`` del results_*.json) y el system el mismo prompt.
    """
    rows = []
    for row in src_rows:
        new = dict(row)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["context"]},
        ]
        t0 = time.perf_counter()
        chat = service._llm.chat_detailed(messages)
        generation_s = round(time.perf_counter() - t0, 3)
        new["answer"] = chat.content
        new["prompt_tokens"] = chat.prompt_tokens
        new["completion_tokens"] = chat.completion_tokens
        new["generation_s"] = generation_s
        new["total_s"] = round(row["retrieval_s"] + generation_s, 3)
        new.pop("error", None)
        rows.append(new)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluacion del pipeline completo (Fase 2A)")
    parser.add_argument("--eval-file", default=str(Path(__file__).resolve().parents[1] / "data" / "eval_questions.json"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--collection", default=None, help="Coleccion Qdrant (por defecto la de .env)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--retrieval-only", action="store_true", help="Saltar la generacion del LLM")
    parser.add_argument("--label", default="baseline", help="Nombre de salida")
    parser.add_argument(
        "--from-results",
        default=None,
        help="Experimento controlado de generacion: reutiliza los chunks/contexto "
        "EXACTOS de un results_*.json previo y solo regenera la respuesta con --model. "
        "El retrieval queda intacto y cambia unicamente el LLM.",
    )
    parser.add_argument("--model", default=None, help="Modelo LLM (por defecto el de .env)")
    args = parser.parse_args()

    if args.from_results and args.retrieval_only:
        parser.error("--from-results no es compatible con --retrieval-only")
    if args.from_results and not args.model:
        parser.error("--from-results requiere --model (modelo con el que regenerar)")

    label = args.label
    settings = get_settings()
    if args.collection:
        settings.collection_name = args.collection
    service = _build_service(settings, args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    from_results_src = None
    if args.from_results:
        from_results_src = json.loads(Path(args.from_results).read_text(encoding="utf-8"))
        try:
            rows = _generate_rows_from_results(service, from_results_src["rows"])
            for row in rows:
                print(f"[{label}] gen={row['generation_s']}s total={row['total_s']}s  {row['question'][:60]}")
        finally:
            service.close()
    else:
        eval_file = Path(args.eval_file)
        data = json.loads(eval_file.read_text(encoding="utf-8"))
        questions = data["questions"]
        book_id = data.get("book_id")
        top_k = args.top_k or data.get("top_k") or 8
        options = BASELINE_OPTIONS
        try:
            rows = []
            for item in questions:
                row = _row(service, item, top_k, book_id, options, with_llm=not args.retrieval_only)
                rows.append(row)
                status = f"recall@8={row['recall@8']:.2f}"
                if row.get("generation_s") is not None:
                    status += f" gen={row['generation_s']}s total={row['total_s']}s"
                elif row.get("error"):
                    status += f" ERROR: {row['error'][:40]}"
                print(f"[{label}] {status}  {row['question'][:60]}")
        finally:
            service.close()

    summary = _metrics(rows)
    if args.from_results:
        src_meta = from_results_src.get("meta") or {}
        meta = {
            "timestamp": timestamp,
            "label": label,
            "llm_model": args.model,
            "llm_think": settings.llm_think,
            "embedding_model": src_meta.get("embedding_model", settings.embedding_model),
            "chunk_tokens": src_meta.get("chunk_tokens", settings.chunk_tokens),
            "chunk_overlap": src_meta.get("chunk_overlap", settings.chunk_overlap),
            "collection": src_meta.get("collection", settings.collection_name),
            "retrieval": f"REUTILIZADA de {Path(args.from_results).name} ({src_meta.get('retrieval', '?')})",
            "top_k": src_meta.get("top_k", settings.top_k),
            "retrieval_only": False,
            "from_results": args.from_results,
            "source_label": src_meta.get("label", "?"),
        }
    else:
        meta = {
            "timestamp": timestamp,
            "label": label,
            "llm_model": settings.llm_model,
            "llm_think": settings.llm_think,
            "embedding_model": settings.embedding_model,
            "chunk_tokens": settings.chunk_tokens,
            "chunk_overlap": settings.chunk_overlap,
            "collection": settings.collection_name,
            "retrieval": "baseline (expansion=off, rerank=none)",
            "top_k": top_k,
            "retrieval_only": args.retrieval_only,
        }
    result = {"meta": meta, "summary": summary, "rows": rows}

    stamp_name = f"results_{label}_{timestamp}.json"
    (out_dir / stamp_name).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / f"results_{label}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = out_dir / f"report_{label}.md"
    _write_report(report_path, meta, rows, summary)

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
                for r in rows
            ],
        }
        (out_dir / f"grades_{label}.json").write_text(
            json.dumps(grades, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print("\n" + "=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"\nJSON:  {out_dir / stamp_name}")
    print(f"Informe: {report_path}")
    print(f"Grades (manual): {out_dir / f'grades_{label}.json'}")


if __name__ == "__main__":
    main()
