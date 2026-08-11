"""LLM via Ollama HTTP (ajuste #6: sin frameworks de IA).

El modelo se configura con LLM_MODEL. Interfaz intercambiable: solo necesita
``chat(messages) -> str``.
"""
import time
from typing import Protocol

import httpx


class LLM(Protocol):
    def chat(self, messages: list[dict]) -> str:
        ...


class OllamaLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 600.0,
        think: bool = False,
        transport: httpx.BaseTransport | None = None,
        attempts: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout
        self._think = think
        self._transport = transport
        self._attempts = attempts

    def chat(self, messages: list[dict]) -> str:
        # think se envia SIEMPRE de forma explicita (false por defecto): no se
        # depende del default de Ollama, que activa el razonamiento en qwen3 y
        # dispara el tiempo por respuesta en hardware modesto.
        body = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": self._think,
            "options": {"temperature": self._temperature},
        }
        for attempt in range(1, self._attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                    resp = client.post(f"{self._base_url}/api/chat", json=body)
                resp.raise_for_status()
                return resp.json()["message"]["content"]
            except (httpx.HTTPError, ValueError, KeyError):
                if attempt == self._attempts:
                    raise
                time.sleep(1.0 * attempt)
        raise RuntimeError("no se pudo generar respuesta")  # pragma: no cover
