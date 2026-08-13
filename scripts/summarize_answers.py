"""Agrega la evaluacion manual de respuestas (Fase 2A).

1) Ejecuta la evaluacion (genera grades_<label>.json):
     python scripts/evaluate_answers.py --collection narrative_c700_o100
2) Rellena a mano por cada pregunta: expected_facts, expected_answer y grade
   (correcta | parcial | incorrecta | sin_evidencia | alucinacion).
3) Agrega el resultado:
     python scripts/summarize_answers.py \
         --results data/eval_answers/results_<label>.json \
         --grades   data/eval_answers/grades_<label>.json

Imprime el recuento por grado y una tabla por pregunta cruzando el grado con
el exito del retrieval (any@8), para distinguir los tipos de fallo:
  - fallo de retrieval   : el capitulo esperado no entro en el top-8.
  - fallo de generacion  : el contexto contenía la evidencia y aun asi la
                           respuesta es incorrecta/parcial.
  - alucinacion          : hechos no respaldados por el contexto.
  - sin_evidencia        : el contexto no bastaba (y el modelo lo declara).

Con --out guarda el resumen en JSON.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

RUBRIC = [
    "correcta",
    "parcial",
    "incorrecta",
    "sin_evidencia",
    "alucinacion",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Agrega la evaluacion manual de respuestas")
    parser.add_argument("--results", required=True, help="results_<label>.json de evaluate_answers.py")
    parser.add_argument("--grades", required=True, help="grades_<label>.json rellenado a mano")
    parser.add_argument("--out", default=None, help="Guardar el resumen en JSON")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    grades = json.loads(Path(args.grades).read_text(encoding="utf-8"))
    rows = results["rows"]
    graded = {g["question"]: g for g in grades["questions"]}

    n = len(rows)
    counts = Counter()
    matrix = []
    missing = []
    for row in rows:
        g = graded.get(row["question"], {})
        grade = g.get("grade")
        retrieval_ok = row.get("retrieval_ok@8", False)
        if not grade:
            missing.append(row["question"])
            continue
        counts[grade] += 1
        if grade == "correcta" and not retrieval_ok:
            kind = "correcta con retrieval fallido (revisar)"
        elif not retrieval_ok:
            kind = "fallo de retrieval"
        elif grade == "incorrecta":
            kind = "fallo de generacion (contexto tenia evidencia)"
        elif grade == "parcial":
            kind = "parcial (respuesta incompleta/ambigua)"
        elif grade == "alucinacion":
            kind = "alucinacion"
        elif grade == "sin_evidencia":
            kind = "sin_evidencia (contexto insuficiente)"
        else:
            kind = "correcta"
        matrix.append(
            (row["question"][:70], retrieval_ok, grade, kind)
        )

    print("=" * 78)
    print(f"EVALUACION MANUAL  ({sum(counts.values())}/{n} preguntas clasificadas)")
    print("=" * 78)
    if counts:
        print("\nRecuento por grado:")
        for g in RUBRIC:
            c = counts.get(g, 0)
            pct = 100 * c / sum(counts.values()) if counts else 0
            print(f"  {g:<14} {c:>3}  ({pct:.0f}%)")

    if matrix:
        print("\nPor pregunta:")
        print(f"{'pregunta':<72} {'ret@8':<6} {'grado':<14} tipo")
        print("-" * 78)
        for q, ok, grade, kind in matrix:
            print(f"{q:<72} {str(ok):<6} {grade:<14} {kind}")
        if sum(counts.values()) < n:
            print(f"\n{len(missing)} preguntas sin clasificar: revisa grades_*.json")

    summary = {
        "total": n,
        "classified": sum(counts.values()),
        "missing": missing,
        "counts": dict(counts),
        "rows": [
            {
                "question": r[0],
                "retrieval_ok@8": r[1],
                "grade": r[2],
                "failure_kind": r[3],
            }
            for r in matrix
        ],
    }
    if args.out:
        Path(args.out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nResumen guardado en: {args.out}")


if __name__ == "__main__":
    main()
