"""Embeddings via Ollama HTTP. Sin frameworks de IA (ajuste #6).

El modelo se configura con EMBEDDING_MODEL y nunca se acopla al LLM.
La dimension del vector se detecta en runtime (no se hardcodea).
"""
import time
from typing import Protocol

import httpx


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OllamaEmbedder:
    def __init__(
        self,
        base_url: str,
        model: str,
        batch_size: int = 32,
        timeout: float = 300.0,
        transport: httpx.BaseTransport | None = None,
        attempts: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._batch_size = batch_size
        self._timeout = timeout
        self._transport = transport
        self._attempts = attempts
        self._dim: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._embed_batch(batch)
            results.extend(embeddings)
        if results and self._dim is None:
            self._dim = len(results[0])
        return results

    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError("dimension desconocida: ejecuta embed() primero")
        return self._dim

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        body = {"model": self._model, "input": batch}
        for attempt in range(1, self._attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                    resp = client.post(f"{self._base_url}/api/embed", json=body)
                resp.raise_for_status()
                data = resp.json()
                return list(data["embeddings"])
            except (httpx.HTTPError, ValueError, KeyError):
                if attempt == self._attempts:
                    raise
                time.sleep(1.0 * attempt)
        raise RuntimeError("no se pudieron generar embeddings")  # pragma: no cover
