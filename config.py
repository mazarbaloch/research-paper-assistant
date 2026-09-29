"""Central configuration. Relative paths resolve against the project, not the shell."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent


class AppError(Exception):
    """An actionable error that can be shown directly in the UI or CLI."""


def env_path(name: str, default: str) -> Path:
    path = Path(os.getenv(name, default)).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


@dataclass(frozen=True)
class Settings:
    ollama_host: str = field(
        default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434")
    )
    generation_model: str = field(
        default_factory=lambda: os.getenv("GENERATION_MODEL", "qwen3.5:9b")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "5")))
    chroma_path: Path = field(default_factory=lambda: env_path("CHROMA_PATH", "data/chroma"))
    papers_path: Path = field(default_factory=lambda: env_path("PAPERS_PATH", "data/papers"))
    collection_name: str = field(
        default_factory=lambda: os.getenv("COLLECTION_NAME", "research_papers")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "64")))
    history_turns: int = field(default_factory=lambda: int(os.getenv("HISTORY_TURNS", "3")))
    ollama_timeout: float = field(default_factory=lambda: float(os.getenv("OLLAMA_TIMEOUT", "180")))
    server_port: int = field(default_factory=lambda: int(os.getenv("GRADIO_SERVER_PORT", "7860")))

    def __post_init__(self):
        if not 0 <= self.chunk_overlap < self.chunk_size or self.chunk_size < 200:
            raise AppError("Use CHUNK_SIZE >= 200 and 0 <= CHUNK_OVERLAP < CHUNK_SIZE.")
        if not 1 <= self.top_k <= 20 or self.batch_size < 1 or self.history_turns < 0:
            raise AppError("Use TOP_K between 1 and 20, BATCH_SIZE >= 1, HISTORY_TURNS >= 0.")
        if self.ollama_timeout <= 0 or not 1 <= self.server_port <= 65535:
            raise AppError("Use OLLAMA_TIMEOUT > 0 and GRADIO_SERVER_PORT between 1 and 65535.")
        host = urlparse(self.ollama_host)
        if host.scheme not in {"http", "https"} or host.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise AppError("OLLAMA_HOST must be a local HTTP address (localhost or loopback).")
        if "cloud" in self.generation_model.lower():
            raise AppError("Choose a locally installed Ollama model, not a cloud model.")


def get_settings() -> Settings:
    try:
        return Settings()
    except ValueError as exc:
        raise AppError("Check numeric configuration values in your environment.") from exc
