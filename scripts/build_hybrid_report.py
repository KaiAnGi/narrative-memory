"""Genera el informe comparativo hybrid (Fase 2C) desde hybrid.json.

Lee el detalle por pregunta del experimento (baseline / narrative / hybrid),
cruza con los grades manuales de Fase 2A (baseline ya rellenado; narrative y
hybrid con skeleton) y escribe data/eval_answers/hybrid_report.md.

Uso:
  python scripts/build_hybrid_report.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "eval_answers"

GRADE_LABELS = {
    "correcta": "Correcta",
    "parcial": "Parcial",
    "incorrecta": "Incorrecta",
    "sin_evidencia": "Sin evidencia",
    "alucinacion": "Alucinacion",
}


def _load(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise SystemExit(f"No existe: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _grade_map(path: Path) -> dict:
    data = _load(path, required=False)
    if not data:
        return {}
    return {q["question"]: q for q in data["questions"]}


def _fmt(row, key, suffix="") -> str:
    v = row.get(key)
    if v is None:
        return "-"
    return f"{v:.3f}{suffix}"


def main() -> None:
    out_dir = DEFAULT_OUT
    result = _load(out_dir / "hybrid.json")
    rows: dict[str, list[dict]] = result["rows"]
    summary = result["summary"]
    meta = result["meta"]

    grades_baseline = _grade_map(out_dir / "grades_baseline700.json")
    grades_narrative = _grade_map(out_dir / "grades_hybrid700_narrative.json")
    grades_hybrid = _grade_map(out_dir / "grades_hybrid700_hybrid.json")

    # ---- Gaps: capítulos esperados que el baseline NO recupera a top-8 ----
    gap_rows = []
    for i, b in enumerate(rows["baseline"]):
        expected = set(b["expected_chapters"])
        got_base = set(b["retrieved_chapters"])
        missing = sorted(expected - got_base)
        if not missing:
            continue
        got_narr = set(rows["narrative"][i]["retrieved_chapters"])
        got_hyb = set(rows["hybrid"][i]["retrieved_chapters"])
        narr_rec = sorted(set(missing) & got_narr)
        hyb_rec = sorted(set(missing) & got_hyb)
        gap_rows.append(
            {
                "question": b["question"],
                "missing": missing,
                "narr_rec": narr_rec,
                "hyb_rec": hyb_rec,
            }
        )

    # ---- Resumen de aportacion de fuentes en hybrid ----
    hsrc = {
        "baseline_only": 0,
        "narrative_only": 0,
        "both": 0,
        "count": 0,
    }
    for r in rows["hybrid"]:
        sc = r.get("source_counts")
        if sc:
            hsrc["baseline_only"] += sc["from_baseline_only"]
            hsrc["narrative_only"] += sc["from_narrative_only"]
            hsrc["both"] += sc["from_both"]
            hsrc["count"] += 1

    # ---- Veredicto preliminar (retrieval) ----
    s = summary
    verdicts = []
    if s["hybrid"]["any_rate@8"] >= s["baseline"]["any_rate@8"]:
        verdicts.append(
            "any@8: el hybrid mantiene (o mejora) la cobertura, sin preguntas sin evidencia"
        )
    if s["hybrid"]["mean_recall@8"] > s["baseline"]["mean_recall@8"]:
        verdicts.append(
            f"recall@8: el hybrid gana {s['hybrid']['mean_recall@8'] - s['baseline']['mean_recall@8']:.3f} "
            "sobre el baseline (mas capitulos esperados en los 8 chunks finales)"
        )
    if s["narrative"]["mean_recall@8"] > s["hybrid"]["mean_recall@8"]:
        verdicts.append(
            f"nota: la memoria sola (narrative) supera al hybrid en recall@8 "
            f"({s['narrative']['mean_recall@8']:.3f} vs {s['hybrid']['mean_recall@8']:.3f}); "
            "la fusion RRF cede algun chunk narrativo a favor de diversidad del baseline"
        )
    if s["hybrid"]["mean_retrieval_s"] > s["baseline"]["mean_retrieval_s"] * 1.5:
        verdicts.append(
            f"coste: el hybrid multiplica x{s['hybrid']['mean_retrieval_s'] / s['baseline']['mean_retrieval_s']:.1f} "
            "el tiempo de retrieval (doble embedding + expansion de capitulos)"
        )

    lines: list[str] = []
    ap = lines.append
    ap("# Informe hybrid: baseline vs memoria narrativa vs hybrid (Fase 2C)\n")
    ap(f"- Fecha: {meta['timestamp']}")
    ap(f"- Etiqueta: `{meta['label']}`")
    ap(f"- Embedder: `{meta['embedding_model']}`")
    ap(f"- LLM: `{meta['llm_model']}`")
    ap(f"- Coleccion: `{meta['collection']}`")
    ap(f"- Memoria: `{meta['memory']}`")
    ap(f"- Presupuesto del LLM: top_k={meta['top_k']} chunks (sin 8+8)")
    ap(f"- Fusion: `{meta['hybrid']}`")
    ap(f"- Retrieval-only: {meta['retrieval_only']}")
    ap("")

    ap("## 1. Metricas de retrieval\n")
    ap("| Metrica | baseline | narrative | hybrid |")
    ap("|---|---:|---:|---:|")
    metric_keys = [
        ("mean_recall@5", "recall@5"),
        ("mean_recall@8", "recall@8"),
        ("any_rate@5", "any@5"),
        ("any_rate@8", "any@8"),
        ("multi_recall@8", "multi-recall@8"),
        ("retrieval_fail_any@8", "preguntas sin evidencia"),
        ("mean_retrieval_s", "tiempo retrieval (s)"),
    ]
    for key, label in metric_keys:
        ap(
            f"| {label} | {s['baseline'].get(key)} | {s['narrative'].get(key)} | "
            f"{s['hybrid'].get(key)} |"
        )
    ap("")

    ap("## 2. Gaps del baseline recuperados por la memoria\n")
    ap(
        "Capitulos esperados que el baseline no llego a recuperar en los 8 chunks "
        "finales, y si la memoria (narrative) o el hybrid los traen de vuelta.\n"
    )
    ap("| Pregunta | Cap. faltantes | narrative | hybrid |")
    ap("|---|---|---|---|")
    for g in gap_rows:
        ap(
            f"| {g['question'][:60]}… | {g['missing']} | {g['narr_rec'] or '-'} | "
            f"{g['hyb_rec'] or '-'} |"
        )
    total_gaps = sum(len(g["missing"]) for g in gap_rows)
    narr_total = sum(len(g["narr_rec"]) for g in gap_rows)
    hyb_total = sum(len(g["hyb_rec"]) for g in gap_rows)
    ap("")
    ap(
        f"- Gaps totales del baseline: **{total_gaps}**; recuperados por narrative: "
        f"**{narr_total}**; por hybrid: **{hyb_total}**."
    )
    ap("")

    ap("## 3. Aportacion de cada fuente en el hybrid\n")
    if hsrc["count"]:
        n = hsrc["count"]
        ap(
            f"De los {n}*8 chunks finales del hybrid: "
            f"{hsrc['baseline_only']} solo del baseline, {hsrc['narrative_only']} solo de la "
            f"memoria, {hsrc['both']} presentes en ambas (deduplicados).\n"
        )
    ap("")

    ap("## 4. Detalle por pregunta (recall@8)\n")
    ap("| # | baseline | narrative | hybrid | Pregunta |")
    ap("|---|---:|---:|---:|---|")
    for i in range(len(rows["baseline"])):
        b = rows["baseline"][i]
        ap(
            f"| Q{i+1} | {_fmt(b, 'recall@8')} | {_fmt(rows['narrative'][i], 'recall@8')} | "
            f"{_fmt(rows['hybrid'][i], 'recall@8')} | {b['question'][:55]}… |"
        )
    ap("")

    ap("## 5. Calidad de las respuestas (grades Fase 2A)\n")
    ap(
        "Clasificacion manual con la rubrica de Fase 2A. El baseline ya esta "
        "rellenado (`grades_baseline700.json`); narrative y hybrid quedan en "
        "`grades_hybrid700_narrative.json` y `grades_hybrid700_hybrid.json`.\n"
    )
    ap("| # | baseline | narrative | hybrid | Pregunta |")
    ap("|---|---|---|---|---|")
    for i, b in enumerate(rows["baseline"]):
        q = b["question"]
        gb = grades_baseline.get(q)
        gn = grades_narrative.get(q)
        gh = grades_hybrid.get(q)
        cell = lambda g: GRADE_LABELS.get(g["grade"], "-") if g and g.get("grade") else "—"
        ap(
            f"| Q{i+1} | {cell(gb)} | {cell(gn)} | {cell(gh)} | {q[:55]}… |"
        )
    ap("")

    ap("## 6. Veredicto (retrieval)\n")
    for v in verdicts:
        ap(f"- {v}")
    ap("")
    ap(
        "**Pendiente:** rellenar los grades de narrative/hybrid (seccion 5) y "
        "`python scripts/summarize_answers.py --labels hybrid700_narrative "
        "hybrid700_hybrid` para el veredicto final de calidad."
    )

    report = "\n".join(lines)
    out_path = out_dir / "hybrid_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Informe: {out_path}")


if __name__ == "__main__":
    main()
