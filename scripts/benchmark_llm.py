"""Benchmark local de modelos LLM via Ollama (diagnostico; no toca produccion).

Mide, para cada modelo y con thinking on/off, usando SIEMPRE la misma pregunta
y el mismo contexto recuperado:
- tiempo hasta el primer token (TTFT)
- tiempo total
- tokens generados y tokens/segundo (eval_count / eval_duration)
- tokens de prompt y tiempo de prefill
- memoria (VRAM exclusiva/compartida y RAM) via /api/ps

Uso:
    python scripts/benchmark_llm.py [--models "qwen3:1.7b qwen3:4b qwen3:8b"]
                                    [--think both] [--runs 2] [--warmup 1]
                                    [--question "..."] [--top-k 8] [--book sample]
                                    [--output data/benchmark_results.json]

Salida: lineas de progreso en stdout + JSON detallado al terminar.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.llm.prompts import build_qa_messages  # noqa: E402
from app.service import Service  # noqa: E402

DEFAULT_QUESTION = (
    "Compara lo que Daniel sabe al final de la historia con lo que sabía al "
    "principio. Cita los capítulos en los que te apoyas."
)


def one_run(base_url: str, model: str, messages: list[dict], think: bool,
            temperature: float, timeout: float = 900.0) -> dict:
    """Una generacion en streaming. Devuelve metricas + contenido."""
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": bool(think),
        "options": {"temperature": temperature},
    }
    t0 = time.perf_counter()
    first_token_s: float | None = None
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    saw_thinking_field = False
    final: dict = {}
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", f"{base_url}/api/chat", json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or line.startswith("data: [DONE]"):
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = chunk.get("message") or {}
                content = msg.get("content") or ""
                thinking = msg.get("thinking") or ""
                if thinking:
                    saw_thinking_field = True
                if (content or thinking) and first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                if content:
                    content_parts.append(content)
                if thinking:
                    thinking_parts.append(thinking)
                if chunk.get("done"):
                    final = chunk

    total_s = time.perf_counter() - t0
    eval_s = final.get("eval_duration", 0) / 1e9
    eval_count = final.get("eval_count")
    tps = round(eval_count / eval_s, 1) if eval_count and eval_s else None
    return {
        "content": "".join(content_parts),
        "thinking_len": sum(len(p) for p in thinking_parts),
        "saw_thinking_field": saw_thinking_field,
        "done_reason": final.get("done_reason"),
        "ttft_s": round(first_token_s, 3) if first_token_s is not None else round(total_s, 3),
        "total_s": round(total_s, 3),
        "prompt_eval_count": final.get("prompt_eval_count"),
        "prompt_eval_s": round(final.get("prompt_eval_duration", 0) / 1e9, 3),
        "eval_count": eval_count,
        "eval_s": round(eval_s, 3),
        "load_s": round(final.get("load_duration", 0) / 1e9, 3),
        "tokens_per_s": tps,
    }


def ps_snapshot(base_url: str) -> list[dict]:
    """Metadatos de memoria de los modelos cargados (VRAM/RAM)."""
    try:
        resp = httpx.get(f"{base_url}/api/ps", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return [
            {
                "model": m["model"],
                "size_gb": round(m["size"] / 1e9, 2),
                "vram_exclusive_gb": round(
                    m.get("size_vram_exclusive", m.get("size_vram", 0)) / 1e9, 2
                ),
                "vram_shared_gb": round(m.get("size_vram_shared", 0) / 1e9, 2),
                "cpu_gb": round(m.get("size_cpu", 0) / 1e9, 2),
                "quantization": (m.get("details") or {}).get("quantization_level"),
            }
            for m in models
        ]
    except Exception:  # noqa: BLE001
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LLM local (Ollama)")
    parser.add_argument("--models", nargs="+", default=["qwen3:1.7b", "qwen3:4b", "qwen3:8b"])
    parser.add_argument("--think", choices=["on", "off", "both"], default="both")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--book", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "data" / "benchmark_results.json"),
    )
    args = parser.parse_args()

    settings = get_settings()
    service = Service(settings)
    try:
        book = args.book or service.last_book_id()
        if book is None:
            print("No hay libro indexado. Ejecuta primero scripts/ingest.py")
            sys.exit(1)
        result = service.search(args.question, top_k=args.top_k, book_id=book)
        hits = result.hits
    finally:
        service.close()

    if not hits:
        print(f"No se recuperaron fragmentos para el libro '{book}'.")
        sys.exit(1)

    messages = build_qa_messages(args.question, hits)
    used_chapters = sorted({h.chunk.chapter_index for h in hits})
    print(f"libro={book} top_k={args.top_k} hits={len(hits)} capitulos={used_chapters}",
          flush=True)

    think_modes = ["off", "on"] if args.think == "both" else [args.think]
    results: dict = {"meta": {}, "runs": []}

    for model in args.models:
        for think in think_modes:
            label = f"{model}:think={think}"
            for _ in range(args.warmup):
                one_run(settings.ollama_base_url, model, messages, think == "on", args.temperature)
            mem = ps_snapshot(settings.ollama_base_url)
            for i in range(args.runs):
                run = one_run(settings.ollama_base_url, model, messages, think == "on",
                              args.temperature)
                entry = {
                    "model": model,
                    "think": think,
                    "run": i + 1,
                    "memory_snapshot": mem,
                    **run,
                }
                results["runs"].append(entry)
                print(
                    f"[{label} run {i + 1}] ttft={entry['ttft_s']}s "
                    f"total={entry['total_s']}s eval_tok={entry['eval_count']} "
                    f"tok/s={entry['tokens_per_s']} prompt={entry['prompt_eval_count']} "
                    f"load={entry['load_s']}s",
                    flush=True,
                )

    results["meta"] = {
        "ollama_base_url": settings.ollama_base_url,
        "question": args.question,
        "book": book,
        "used_chapters": used_chapters,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "runs_per_config": args.runs,
        "warmup": args.warmup,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK guardado en {out}", flush=True)


if __name__ == "__main__":
    main()
