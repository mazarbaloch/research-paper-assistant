from dataclasses import replace

import pytest

import retrieval
from config import AppError, Settings
from retrieval import records_from_chroma, retrieve
from storage import open_collection


def test_record_transformation():
    records = records_from_chroma(
        {
            "ids": [["paper-page-6-chunk-3"]],
            "documents": [["An exact extracted passage."]],
            "metadatas": [[{"title": "Paper", "filename": "paper.pdf", "page": 6}]],
            "distances": [[0.18]],
        }
    )
    assert records[0].page == 6
    assert records[0].text == "An exact extracted passage."
    assert records[0].distance == 0.18
    assert records_from_chroma({"ids": [[]]}) == []


def test_real_persistent_chroma_query_and_upsert(tmp_path, monkeypatch):
    # Explicit numeric unit vectors test database semantics; not a fake research corpus.
    settings = replace(Settings(), chroma_path=tmp_path / "chroma")
    collection = open_collection(settings, create=True)
    rows = dict(
        ids=["unit-a", "unit-b"],
        documents=["Fixture A", "Fixture B"],
        metadatas=[
            {"paper_id": "a", "title": "A", "filename": "a.pdf", "page": 2},
            {"paper_id": "b", "title": "B", "filename": "b.pdf", "page": 8},
        ],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    collection.upsert(**rows)
    collection.upsert(**rows)
    assert (settings.chroma_path / "chroma.sqlite3").is_file()
    reopened = open_collection(settings)
    assert reopened.count() == 2
    assert reopened.configuration["hnsw"]["space"] == "cosine"
    monkeypatch.setattr(retrieval, "embed", lambda texts, config: [[1.0, 0.0, 0.0]])
    records = retrieve("Unit query", settings, top_k=5)
    assert len(records) == 2
    assert records[0].id == "unit-a"
    assert records[0].distance == pytest.approx(0.0)
    with pytest.raises(AppError, match="configuration differs"):
        open_collection(replace(settings, embedding_model="different-model"))


def test_missing_index_does_not_create_storage(tmp_path):
    settings = replace(Settings(), chroma_path=tmp_path / "absent")
    with pytest.raises(AppError, match="uv run python ingest.py"):
        retrieve("question", settings)
    assert not settings.chroma_path.exists()


def test_blank_question():
    with pytest.raises(AppError, match="Enter a research question"):
        retrieve("   ", Settings())
