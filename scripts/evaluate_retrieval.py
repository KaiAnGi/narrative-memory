"""Evaluacion del retrieval (ajuste #7).

Mide, para cada pregunta de un archivo JSON, que capitulos esperados
aparecen entre los top-k recuperados. Los tests unitarios verifican que el
codigo funciona; esto mide la CALIDAD de la recuperacion.

Formato del archivo JSON (por defecto data/eval_questions.json):

{
  "book_id": "sample",
  "top_k": 8,
  "questions": [
    {"question": "...", "expected_chapters": [2, 3]},
    ...
  ]
}

Uso: python scripts/evaluate_retrieval.py [--eval-file ruta.json] [--top-k N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.service import Service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalúa la recuperación")
    parser.add_argument("--eval-file", default=str(Path(__file__).resolve().parents[1] / "data" / "eval_questions.json"))
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    eval_file = Path(args.eval_file)
    data = json.loads(eval_file.read_text(encoding="utf-8"))
    top_k = args.top_k or data.get("top_k") or 8

    service = Service(get_settings())
    try:
        default_book = data.get("book_id") or service.last_book_id()
        _run_evaluation(service, data, default_book, top_k, eval_file)
    finally:
        service.close()


def _run_evaluation(service, data, default_book, top_k, eval_file):
    rows = []
    recall_sum = 0.0
    any_hit = 0
    for item in data["questions"]:
        question = item["question"]
        expected = set(item.get("expected_chapters", []))
        result = service.search(
            question,
            top_k=top_k,
            book_id=item.get("book_id", default_book),
        )
        got = {h.chunk.chapter_index for h in result.hits}
        recall = len(expected & got) / len(expected) if expected else 0.0
        hit = bool(expected & got)
        recall_sum += recall
        any_hit += int(hit)
        rows.append(
            {
                "question": question,
                "expected_chapters": sorted(expected),
                "retrieved_chapters": sorted(got),
                "recall_at_k": round(recall, 3),
                "any_expected_chapter_retrieved": hit,
            }
        )
        print(f"[{'OK ' if hit else 'FALLO'}] esperados={sorted(expected)} recuperados={sorted(got)} recall@{top_k}={recall:.2f}")
        print(f"      {question}")

    n = len(rows)
    summary = {
        "top_k": top_k,
        "questions": n,
        "mean_recall_at_k": round(recall_sum / n, 3) if n else 0.0,
        "any_expected_chapter_rate": round(any_hit / n, 3) if n else 0.0,
    }
    print("\n" + "=" * 70)
    print(f"RESUMEN  recall@{top_k} medio={summary['mean_recall_at_k']:.3f}  "
          f"tasa de acierto (>=1 capítulo esperado)={summary['any_expected_chapter_rate']:.3f}")

    out = eval_file.with_name(f"{eval_file.stem}.results.json")
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Resultados detallados guardados en: {out}")


if __name__ == "__main__":
    main()
