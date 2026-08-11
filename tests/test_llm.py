"""Tests de OllamaLLM: think se envia SIEMPRE explicito (false por defecto)."""
import json

import httpx
import pytest

from app.core.config import Settings
from app.llm.ollama_llm import OllamaLLM


def _capture_chat() -> tuple[list[dict], httpx.MockTransport]:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={"message": {"role": "assistant", "content": "RESPUESTA FAKE"}},
            )
        return httpx.Response(404)

    return bodies, httpx.MockTransport(handler)


def test_chat_sends_think_false_by_default():
    bodies, transport = _capture_chat()
    llm = OllamaLLM(base_url="http://ollama", model="qwen3:1.7b", transport=transport)
    assert llm.chat([{"role": "user", "content": "hola"}]) == "RESPUESTA FAKE"

    body = bodies[0]
    assert body["think"] is False
    assert body["model"] == "qwen3:1.7b"


def test_chat_sends_think_true_when_configured():
    bodies, transport = _capture_chat()
    llm = OllamaLLM(
        base_url="http://ollama",
        model="qwen3:8b",
        think=True,
        transport=transport,
    )
    llm.chat([{"role": "user", "content": "razona"}])
    assert bodies[0]["think"] is True


def test_settings_default_model_and_think():
    settings = Settings(_env_file=None)
    assert settings.llm_model == "qwen3:1.7b"
    assert settings.llm_think is False


@pytest.mark.parametrize("env_value,expected", [("true", True), ("false", False), ("1", True)])
def test_settings_llm_think_from_env(monkeypatch, env_value, expected):
    monkeypatch.setenv("LLM_THINK", env_value)
    settings = Settings(_env_file=None)
    assert settings.llm_think is expected
