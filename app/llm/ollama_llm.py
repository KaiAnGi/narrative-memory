"""LLM via Ollama HTTP (ajuste #6: sin frameworks de IA).

El modelo se configura con LLM_MODEL. Interfaz intercambiable: solo necesita
``chat(messages) -> str``. ``chat_detailed`` ademas devuelve el conteo de
tokens (prompt/completion) cuando Ollama los proporciona.
"""
import time
from dataclasses import dataclass
from typing import Protocol

import httpx


class LLM(Protocol):
    def chat(self, messages: list[dict]) -> str:
        ...


@dataclass(frozen=True)
class ChatResult:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


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
        return self.chat_detailed(messages).content

    def chat_detailed(self, messages: list[dict]) -> ChatResult:
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
                data = resp.json()
                return ChatResult(
                    content=data["message"]["content"],
                    prompt_tokens=_tokens(data, "prompt_eval_count"),
                    completion_tokens=_tokens(data, "eval_count"),
                )
            except (httpx.HTTPError, ValueError, KeyError):
                if attempt == self._attempts:
                    raise
                time.sleep(1.0 * attempt)
        raise RuntimeError("no se pudo generar respuesta")  # pragma: no cover


def _tokens(data: dict, key: str) -> int | None:
    """Lee el conteo de tokens de la respuesta de /api/chat.

    Ollama lo expone en la raiz (prompt_eval_count/eval_count) o anidado en
    "usage" (prompt_tokens/completion_tokens) segun version.
    """
    if data.get(key) is not None:
        return int(data[key])
    usage = data.get("usage") or {}
    mapped = {"prompt_eval_count": "prompt_tokens", "eval_count": "completion_tokens"}
    value = usage.get(mapped.get(key))
    return int(value) if value is not None else None
