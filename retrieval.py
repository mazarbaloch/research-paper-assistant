"""Semantic retrieval only: no generation, implicit embeddings, or fabricated results."""

from config import AppError, Settings
from models import Passage
from storage import embed, open_collection


def records_from_chroma(result: dict) -> list[Passage]:
    if not result.get("ids") or not result["ids"][0]:
        return []
    return [
        Passage(
            id=chunk_id,
            text=text,
            title=metadata["title"],
            filename=metadata["filename"],
            page=metadata["page"],
            distance=distance,
        )
        for chunk_id, text, metadata, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
    ]


def retrieve(question: str, settings: Settings, top_k: int | None = None) -> list[Passage]:
    if not question.strip():
        raise AppError("Enter a research question first.")
    count = settings.top_k if top_k is None else top_k
    if not 1 <= count <= 20:
        raise AppError("Choose between 1 and 20 retrieved passages.")
    collection = open_collection(settings)
    available = collection.count()
    if not available:
        raise AppError("No indexed research papers were found. Run: uv run python ingest.py")
    query_embedding = embed([question], settings)
    try:
        result = collection.query(
            query_embeddings=query_embedding,
            n_results=min(count, available),
            include=["documents", "metadatas", "distances"],
        )
        return records_from_chroma(result)
    except Exception as exc:
        raise AppError(
            "Could not search the index. Check Chroma access and re-run ingestion."
        ) from exc
