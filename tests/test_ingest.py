from dataclasses import replace

import pytest

import ingest as ingestion
from config import AppError, Settings
from models import Chunk
from storage import open_collection


def test_missing_and_empty_paper_directory(tmp_path):
    settings = replace(Settings(), papers_path=tmp_path / "papers")
    with pytest.raises(AppError, match="directory missing"):
        ingestion.ingest(settings)
    settings.papers_path.mkdir()
    with pytest.raises(AppError, match="No PDF files"):
        ingestion.ingest(settings)


def test_incremental_index_and_replacement(tmp_path, monkeypatch):
    settings = replace(Settings(), papers_path=tmp_path / "papers", chroma_path=tmp_path / "db")
    settings.papers_path.mkdir()
    path = settings.papers_path / "unit.pdf"
    # These bytes are a mocked unit-test boundary, NOT a generated/fake research PDF.
    path.write_bytes(b"unit fixture version one")
    identity = ingestion.paper_identity("unit.pdf")
    chunks = [
        Chunk(
            id=ingestion.chunk_id(identity, 1, i),
            text=f"unit {i}",
            paper_id=identity,
            title="Unit fixture",
            filename="unit.pdf",
            page=1,
            chunk_index=i,
        )
        for i in range(2)
    ]
    calls = []
    monkeypatch.setattr(ingestion, "extract_chunks", lambda *args: (chunks, 1, 0))

    def unit_vectors(texts, config):
        calls.append(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(ingestion, "embed", unit_vectors)
    first = ingestion.ingest(settings, lambda message: None)
    second = ingestion.ingest(settings, lambda message: None)
    assert first["chunks"] == second["chunks"] == 2
    assert second["skipped"] == 1
    assert len(calls) == 1
    monkeypatch.setattr(ingestion, "EXTRACTION_VERSION", "updated-extractor")
    migrated = ingestion.ingest(settings, lambda message: None)
    assert migrated["indexed"] == 1 and migrated["chunks"] == 2
    assert len(calls) == 2
    path.write_bytes(b"unit fixture version two")
    chunks.pop()
    third = ingestion.ingest(settings, lambda message: None)
    assert third["chunks"] == 1
    assert len(calls) == 3
    assert open_collection(settings).get()["ids"] == [chunks[0].id]


def test_unreadable_pdf_is_actionable(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not a PDF")
    with pytest.raises(AppError, match="could not read this PDF"):
        ingestion.extract_chunks(path, replace(Settings(), papers_path=tmp_path))
