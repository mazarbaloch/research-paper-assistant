import pytest

from ingest import chunk_id, chunk_text, clean_text, paper_identity


def test_empty_page():
    assert chunk_text(" \n\t\x00 ") == []


def test_short_page_and_cleanup():
    assert chunk_text("An ex-\nample.\n\nAnother paragraph.") == ["An example. Another paragraph."]
    assert clean_text("hy\u00adphen") == "hyphen"


def test_chunks_are_bounded_overlap_and_cover_words():
    text = " ".join(f"word{index:04d}" for index in range(700))
    chunks = chunk_text(text, size=1000, overlap=150)
    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 1000 for chunk in chunks)
    for left, right in zip(chunks, chunks[1:]):
        assert any(left.endswith(right[:length]) for length in range(140, 180))
    assert set(text.split()) == set(" ".join(chunks).split())


def test_long_unbroken_text_terminates():
    chunks = chunk_text("x" * 4001, 1000, 150)
    assert len(chunks) == 5
    assert all(len(chunk) <= 1000 for chunk in chunks)


def test_stable_ids_and_page_provenance():
    identity = paper_identity("folder/paper.pdf")
    assert identity == paper_identity("folder/paper.pdf")
    assert identity != paper_identity("other/paper.pdf")
    assert chunk_id(identity, 6, 3) == f"{identity}-page-6-chunk-3"
    assert chunk_id(identity, 6, 3) != chunk_id(identity, 7, 3)


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_config(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("input", size, overlap)
