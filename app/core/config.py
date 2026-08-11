"""Configuracion centralizada. Toda URL/modelo/path vive aqui, nunca hardcodeado."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "narrative-memory"
    version: str = "0.1.0"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:1.7b"
    # think=false es la configuracion recomendada para hardware modesto (4 GB VRAM):
    # con modelos "thinking" (qwen3) el razonamiento oculto duplica/tiplica el tiempo
    # por respuesta. Puede activarse (LLM_THINK=true) para preguntas que exijan razonar.
    llm_think: bool = False
    embedding_model: str = "qwen3-embedding:0.6b"

    # Qdrant
    qdrant_mode: str = "local"  # "local" | "remote"
    qdrant_url: str = "http://localhost:6333"
    qdrant_local_path: Path = BASE_DIR / "data" / "qdrant_local"
    collection_name: str = "narrative_chunks"

    # Datos
    books_dir: Path = BASE_DIR / "data" / "books"

    # Ingestion / retrieval (ajuste #2: valores iniciales, no definitivos)
    chunk_tokens: int = 500
    chunk_overlap: int = 50
    top_k: int = 8
    embedding_batch_size: int = 32
    llm_temperature: float = 0.2
    embedding_timeout_seconds: float = 300.0
    llm_timeout_seconds: float = 600.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
