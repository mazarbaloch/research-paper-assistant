"""Persistent storage and explicit, shared embedding operations (loaded on demand)."""

import hashlib
import json
from functools import lru_cache

from config import AppError, Settings


def index_signature(settings: Settings) -> str:
    payload = [settings.embedding_model, settings.chunk_size, settings.chunk_overlap, "v1"]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def open_collection(settings: Settings, *, create: bool = False):
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    if not create and not (settings.chroma_path / "chroma.sqlite3").exists():
        raise AppError(
            "No indexed research papers were found. Add PDFs to "
            f"{settings.papers_path} and run: uv run python ingest.py"
        )
    try:
        client = chromadb.PersistentClient(
            path=str(settings.chroma_path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        if create:
            collection = client.get_or_create_collection(
                settings.collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={"index_signature": index_signature(settings)},
            )
        else:
            from chromadb.errors import NotFoundError

            try:
                collection = client.get_collection(
                    settings.collection_name, embedding_function=None
                )
            except NotFoundError as exc:
                raise AppError("Collection is missing. Run: uv run python ingest.py") from exc
        if (collection.metadata or {}).get("index_signature") != index_signature(settings):
            raise AppError(
                "Index configuration differs from the stored index. Restore the original "
                "embedding/chunk settings or use a new COLLECTION_NAME and run ingestion."
            )
        return collection
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            f"Could not open Chroma at {settings.chroma_path}. Check directory permissions "
            "and available disk space."
        ) from exc


@lru_cache(maxsize=2)
def embedding_model(name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed(texts: list[str], settings: Settings) -> list[list[float]]:
    try:
        # Both documents and queries use exactly this explicit normalized encoding path.
        return (
            embedding_model(settings.embedding_model)
            .encode(
                texts,
                normalize_embeddings=True,
                batch_size=settings.batch_size,
                show_progress_bar=False,
            )
            .tolist()
        )
    except Exception as exc:
        raise AppError(
            f"Could not load or run embedding model {settings.embedding_model}. "
            "First use requires internet access to download the model; later runs use the cache. "
            "Check the model name, cache access, and available memory."
        ) from exc


def index_stats(settings: Settings) -> tuple[int, int]:
    if not (settings.chroma_path / "chroma.sqlite3").exists():
        return 0, 0
    collection = open_collection(settings)
    count = collection.count()
    paper_ids = set()
    for offset in range(0, count, 1000):
        rows = collection.get(limit=1000, offset=offset, include=["metadatas"])
        paper_ids.update(meta["paper_id"] for meta in rows["metadatas"])
    return len(paper_ids), count
