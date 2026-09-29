from dataclasses import replace

import app
import ingest
import storage
from app import build_app, render_answer, retrieval_question, source_files
from config import Settings
from models import GroundedAnswer, Passage


def test_app_build_does_not_ingest_or_embed(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Startup must not ingest or embed")

    monkeypatch.setattr(ingest, "ingest", forbidden)
    monkeypatch.setattr(storage, "embed", forbidden)
    settings = replace(Settings(), chroma_path=tmp_path / "absent")
    demo = build_app(settings)
    assert demo.title == "Research Paper Knowledge Assistant"
    assert not settings.chroma_path.exists()


def test_sources_retain_numbers_and_escape_pdf_content():
    passages = [
        Passage(
            id=str(i),
            text="<script>bad()</script>",
            title="<Paper>",
            filename="file.pdf",
            page=i,
            distance=0.2,
        )
        for i in (1, 2)
    ]
    rendered = render_answer(
        GroundedAnswer(answer="Claim [2]", evidence_status="supported", citations=[2]), passages
    )
    assert "[2] &lt;Paper&gt; — Page 2" in rendered
    assert "[1]" not in rendered
    assert "<script>" not in rendered


def test_pdf_download_cannot_escape_paper_directory(tmp_path):
    outside = tmp_path / "private.pdf"
    outside.write_bytes(b"unit fixture")
    root = tmp_path / "papers"
    root.mkdir()
    passage = Passage(
        id="unit", text="unit", title="unit", filename="../private.pdf", page=1, distance=0
    )
    assert source_files([passage], [1], replace(Settings(), papers_path=root)) == []


def test_retrieval_context_excludes_previous_model_answers():
    result = retrieval_question(
        "Which of those?",
        [
            {"role": "user", "content": "Retrieval limitations?"},
            {"role": "assistant", "content": "An old unsupported model claim"},
        ],
    )
    assert "Retrieval limitations?" in result
    assert "unsupported model claim" not in result


def test_standalone_question_does_not_inherit_unrelated_topics():
    question = "How is retrieval evaluated?"
    assert (
        retrieval_question(
            question,
            [
                {"role": "user", "content": "Which papers discuss hallucinations?"},
                {"role": "assistant", "content": "An older answer [7]."},
            ],
        )
        == question
    )


def test_new_topic_clears_hidden_context_for_subsequent_followup(tmp_path, monkeypatch):
    queries = []
    contexts = []
    record = Passage(
        id="unit",
        text="Unit fixture",
        title="Unit paper",
        filename="unit.pdf",
        page=1,
        distance=0.1,
    )

    def search(query, *args):
        queries.append(query)
        return [record]

    def answer(question, passages, settings, history):
        contexts.append(list(history))
        return GroundedAnswer(answer="Unit claim [1]", evidence_status="supported", citations=[1])

    monkeypatch.setattr(app, "retrieve", search)
    monkeypatch.setattr(app, "generate_answer", answer)
    settings = replace(Settings(), papers_path=tmp_path)
    _, chat, history, _ = app.ask("Which papers discuss reranking?", [], [], 5, settings)
    _, chat, history, _ = app.ask("How is retrieval evaluated?", chat, history, 5, settings)
    assert len(chat) == 4  # Old messages remain visible.
    assert len(history) == 2  # Only the new topic is conversational context.
    app.ask("Which paper supports that explanation?", chat, history, 5, settings)
    assert "reranking" not in queries[-1]
    assert "How is retrieval evaluated?" in queries[-1]
    assert contexts[1] == []
    assert all("reranking" not in item["content"] for item in contexts[-1])
