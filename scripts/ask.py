"""Hace una pregunta sobre la novela indexada y muestra los fragmentos recuperados.

Uso: python scripts/ask.py "pregunta" [--top-k N] [--chapter N] [--book ID] [--no-hits]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.service import Service  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pregunta sobre la novela (RAG)")
    parser.add_argument("question", help="La pregunta en lenguaje natural")
    parser.add_argument("--top-k", type=int, default=None, help="Número de fragmentos a recuperar")
    parser.add_argument("--chapter", type=int, default=None, help="Restringir a un capítulo")
    parser.add_argument("--book", type=str, default=None, help="book_id (por defecto: el último indexado)")
    parser.add_argument("--no-hits", action="store_true", help="No mostrar los fragmentos recuperados")
    args = parser.parse_args()

    service = Service(get_settings())
    try:
        book = args.book
        if book is None:
            book = service.last_book_id()
        answer = service.ask_question(
            args.question,
            top_k=args.top_k,
            book_id=book,
            chapter=args.chapter,
        )
    finally:
        service.close()

    if not args.no_hits:
        print("=" * 70)
        print("FRAGMENTOS RECUPERADOS")
        for i, hit in enumerate(answer.hits, start=1):
            chunk = hit.chunk
            snippet = chunk.text.replace("\n", " ")[:160]
            print(f"  [{i}] Capítulo {chunk.chapter_index} · posición {chunk.global_position}"
                  f" · score {hit.score:.3f}")
            print(f"      {snippet}...")
        print("=" * 70)

    print(f"\nPREGUNTA: {answer.question}")
    print(f"CAPÍTULOS USADOS: {answer.used_chapters}")
    print("\nRESPUESTA:\n")
    print(answer.answer)


if __name__ == "__main__":
    main()
