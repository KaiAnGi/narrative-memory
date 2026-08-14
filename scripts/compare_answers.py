"""Compara dos ejecuciones de generacion de respuestas (Fase 2D).

Cruce por pregunta de dos pares (results_*.json + grades_*.json): grade,
tiempos y tokens de generacion, con resumen global. Pensado para comparar
variantes de prompt con el MISMO contexto recuperado (p. ej. baseline vs
grounding), pero sirve para cualquier par de ejecuciones.

Uso:
  python scripts/compare_answers.py \
      --a-results data/eval_answers/results_baseline700.json \
      --a-grades  data/eval_answers/grades_baseline700.json \
      --a-label baseline \
      --b-results data/eval_answers/results_grounding700.json \
      --b-grades  data/eval_answers/grades_grounding700.json \
      --b-label grounding \
      --out data/eval_answers/compare_grounding_vs_baseline.md

Genera una tabla por pregunta (grade A vs B, tiempo y tokens) y un resumen
global (correctas, parciales, incorrectas, alucinaciones, sin evidencia, tiempo
y tokens medios), mas las respuestas completas de las preguntas cuyo grade
cambio, para revisarlas sin abrir los JSON.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RUBRIC = [
    "correcta",
    "parcial",
    "incorrecta",
    "sin_evidencia",
    "alucinacion",
]

GRADE_ORDER = {g: i for i, g in enumerate(RUBRIC)}


def _load(path: str, label: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[{label}] No existe: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _grade_map(grades: dict) -> dict:
    return {q["question"]: q.get("grade") for q in grades["questions"]}


def _grade_counts(grades: dict) -> Counter:
    return Counter(g.get("grade") for g in grades["questions"] if g.get("grade"))


def _mean(rows: list[dict], key: str) -> float | None:
    vals = [r.get(key) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 1)


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.1f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara dos ejecuciones de generacion")
    parser.add_argument("--a-results", required=True)
    parser.add_argument("--a-grades", required=True)
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-results", required=True)
    parser.add_argument("--b-grades", required=True)
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--out", required=True, help="Ruta del informe markdown")
    args = parser.parse_args()

    a = _load(args.a_results, args.a_label)
    b = _load(args.b_results, args.b_label)
    grades_a = _grade_map(_load(args.a_grades, args.a_label))
    grades_b = _grade_map(_load(args.b_grades, args.b_label))
    rows_a = {r["question"]: r for r in a["rows"]}
    rows_b = {r["question"]: r for r in b["rows"]}

    meta_a, meta_b = a.get("meta", {}), b.get("meta", {})
    prompt_a = meta_a.get("prompt") or args.a_label
    prompt_b = meta_b.get("prompt") or args.b_label
    model_a = meta_a.get("llm_model", "?")
    model_b = meta_b.get("llm_model", "?")

    questions = [r["question"] for r in a["rows"]]
    counts_a = Counter(grades_a.get(q) for q in questions if grades_a.get(q))
    counts_b = Counter(grades_b.get(q) for q in questions if grades_b.get(q))
    n_classified_a = sum(counts_a.values())
    n_classified_b = sum(counts_b.values())

    print(f"\n{'=' * 78}")
    print(f"COMPARACION: {args.a_label} ({prompt_a}) vs {args.b_label} ({prompt_b})")
    print(f"{'=' * 78}")
    print("\nResumen global:")
    print(f"{'grade':<14} {args.a_label:>12} {args.b_label:>12}   cambio")
    for g in RUBRIC:
        ca, cb = counts_a.get(g, 0), counts_b.get(g, 0)
        delta = "+" + str(cb - ca) if cb > ca else str(cb - ca)
        print(f"{g:<14} {ca:>12} {cb:>12}   {delta}")
    ra = _mean(a["rows"], "generation_s")
    rb = _mean(b["rows"], "generation_s")
    ta = _mean(a["rows"], "completion_tokens")
    tb = _mean(b["rows"], "completion_tokens")
    print(f"{'tiempo gen medio':<14} {_fmt(ra):>12} {_fmt(rb):>12}")
    print(f"{'completion tok medio':<14} {_fmt(ta):>12} {_fmt(tb):>12}")

    lines = [
        "# Comparacion de generacion (Fase 2D)",
        "",
        f"**{args.a_label}**: modelo `{model_a}` · prompt `{prompt_a}` · "
        f"think={meta_a.get('llm_think')}",
        f"**{args.b_label}**: modelo `{model_b}` · prompt `{prompt_b}` · "
        f"think={meta_b.get('llm_think')}",
        "",
        "Mismo contexto de retrieval en ambas ejecuciones (reutilizado).",
        "",
        "## Resumen global",
        "",
        "| grade | " + args.a_label + " | " + args.b_label + " | cambio |",
        "|---|---:|---:|---:|",
    ]
    for g in RUBRIC:
        ca, cb = counts_a.get(g, 0), counts_b.get(g, 0)
        delta = f"+{cb - ca}" if cb > ca else str(cb - ca)
        lines.append(f"| {g} | {ca} | {cb} | {delta} |")
    lines += [
        "",
        "| metrica | " + args.a_label + " | " + args.b_label + " |",
        "|---|---:|---:|",
        f"| tiempo generacion medio (s) | {_fmt(ra)} | {_fmt(rb)} |",
        f"| tiempo total medio (s) | {_fmt(_mean(a['rows'], 'total_s'))} | "
        f"{_fmt(_mean(b['rows'], 'total_s'))} |",
        f"| prompt tokens medio | {_fmt(_mean(a['rows'], 'prompt_tokens'))} | "
        f"{_fmt(_mean(b['rows'], 'prompt_tokens'))} |",
        f"| completion tokens medio | {_fmt(ta)} | {_fmt(tb)} |",
        f"| preguntas clasificadas | {n_classified_a}/{len(a['rows'])} | "
        f"{n_classified_b}/{len(b['rows'])} |",
        "",
        "## Por pregunta",
        "",
        f"| # | pregunta | ret@8 | {args.a_label} | {args.b_label} | gen A (s) | "
        f"gen B (s) | comp A | comp B |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    changed = []
    for i, q in enumerate(questions, start=1):
        ga = grades_a.get(q)
        gb = grades_b.get(q)
        ra_ = rows_a.get(q, {}).get("generation_s")
        rb_ = rows_b.get(q, {}).get("generation_s")
        ta_ = rows_a.get(q, {}).get("completion_tokens")
        tb_ = rows_b.get(q, {}).get("completion_tokens")
        if ga != gb:
            changed.append(q)
        ret = rows_a.get(q, {}).get("retrieval_ok@8")
        lines.append(
            f"| {i} | {q[:55]}… | {ret} | {ga or '—'} | {gb or '—'} | {_fmt(ra_)} | "
            f"{_fmt(rb_)} | {_fmt(ta_)} | {_fmt(tb_)} |"
        )

    lines += ["", "## Preguntas con grade distinto", ""]
    if not changed:
        lines.append("Ninguna pregunta cambia de grade entre ambas ejecuciones.")
    for q in changed:
        ga, gb = grades_a.get(q), grades_b.get(q)
        lines.append(f"### {q}")
        lines.append("")
        lines.append(f"- {args.a_label}: **{ga}**  |  {args.b_label}: **{gb}**")
        lines.append("")
        lines.append(f"<details><summary>{args.a_label} — respuesta</summary>")
        lines.append("")
        lines.append(rows_a.get(q, {}).get("answer") or "(sin respuesta)")
        lines.append("")
        lines.append("</details>")
        lines.append("")
        lines.append(f"<details><summary>{args.b_label} — respuesta</summary>")
        lines.append("")
        lines.append(rows_b.get(q, {}).get("answer") or "(sin respuesta)")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nInforme: {out}")


if __name__ == "__main__":
    main()
