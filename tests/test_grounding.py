from types import SimpleNamespace

import httpx
import ollama
import pytest
from pydantic import ValidationError

import ai_service
from ai_service import generate_answer, generation_schema, render_draft, validate_answer
from config import AppError, Settings
from models import AnswerDraft, CitedParagraph, GroundedAnswer, Passage


def draft(text="In {paper:1}, the authors discuss a claim.", citations=None):
    return AnswerDraft(
        evidence_status="supported",
        paragraphs=[CitedParagraph(text=text, citations=[1] if citations is None else citations)],
    )


def supported(text="A claim [1].", citations=None):
    return GroundedAnswer(
        answer=text, evidence_status="supported", citations=[1] if citations is None else citations
    )


def test_valid_citations():
    assert validate_answer(supported("A [2]. B [1] [2].", [2, 1]), 2).citations == [2, 1]


@pytest.mark.parametrize(
    "answer",
    [
        supported("A [7].", [7]),
        supported("A [0].", [0]),
        supported("A [2].", [1]),
        supported("No citation.", [1]),
        supported("A [1].", []),
        supported("A [1, 2].", [1, 2]),
        supported("A [1-2].", [1, 2]),
        supported("A [1].", [1, 1]),
    ],
)
def test_invalid_citations_rejected(answer):
    with pytest.raises(ValueError):
        validate_answer(answer, 2)


def test_insufficient_never_leaks_unsupported_prose():
    answer = GroundedAnswer(answer="Unsupported model prose", evidence_status="insufficient")
    validated = validate_answer(answer, 5)
    assert validated.citations == []
    assert "Unsupported model prose" not in validated.answer


def test_insufficient_cannot_cite_similar_passages():
    with pytest.raises(ValueError):
        validate_answer(
            GroundedAnswer(answer="Maybe [1]", evidence_status="insufficient", citations=[1]), 5
        )


def test_empty_evidence_never_calls_model(monkeypatch):
    monkeypatch.setattr(ai_service.ollama, "Client", lambda **kw: pytest.fail("Called model"))
    assert generate_answer("Question", [], Settings()).evidence_status == "insufficient"


def test_pydantic_rejects_noninteger_citations():
    with pytest.raises(ValidationError):
        GroundedAnswer.model_validate_json(
            '{"answer":"A [1]", "evidence_status":"supported", "citations":["1"]}'
        )


@pytest.fixture
def passage():
    # An explicit unit fixture, never ingested as a research paper.
    return Passage(
        id="unit",
        text="Unit-test passage",
        title="Unit fixture",
        filename="unit.pdf",
        page=6,
        distance=0.2,
    )


class StubClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return SimpleNamespace(message=SimpleNamespace(content=result))


def test_retry_and_fresh_evidence(monkeypatch, passage):
    client = StubClient(['{"bad":true}', draft().model_dump_json()])
    monkeypatch.setattr(ai_service.ollama, "Client", lambda **kw: client)
    answer = generate_answer(
        "Question",
        [passage],
        Settings(),
        [{"role": "assistant", "content": "Old unsupported answer"}],
    )
    assert answer.citations == [1]
    assert len(client.calls) == 2
    prompt = client.calls[0]["messages"][1]["content"]
    assert "conversation_context_not_evidence" in prompt
    assert "fresh_retrieved_evidence" in prompt
    assert "Page: 6" in prompt
    assert client.calls[0]["format"] == generation_schema(1)
    repair_messages = client.calls[1]["messages"]
    assert repair_messages[-2]["role"] == "assistant"
    assert "Validation error:" in repair_messages[-1]["content"]


def test_repeated_bad_output_is_not_displayed(monkeypatch, passage):
    client = StubClient([draft("Claim", [7]).model_dump_json()] * 2)
    monkeypatch.setattr(ai_service.ollama, "Client", lambda **kw: client)
    with pytest.raises(AppError, match="No unverified answer"):
        generate_answer("Question", [passage], Settings())


@pytest.mark.parametrize(
    "error, message",
    [
        (ConnectionError(), "Make sure Ollama is running"),
        (httpx.ReadTimeout("timeout"), "OLLAMA_TIMEOUT"),
        (ollama.ResponseError("missing", 404), "ollama pull qwen3.5:9b"),
    ],
)
def test_actionable_ollama_errors(monkeypatch, passage, error, message):
    monkeypatch.setattr(ai_service.ollama, "Client", lambda **kw: StubClient([error]))
    with pytest.raises(AppError, match=message):
        generate_answer("Question", [passage], Settings())


def test_application_renders_citations_without_requiring_model_inline_markers(passage):
    answer = render_draft(draft(), [passage])
    assert answer.answer == "In “Unit fixture”, the authors discuss a claim. [1]"
    assert answer.citations == [1]
    assert "{paper:" not in answer.answer


def test_redundant_valid_citations_are_canonicalized(passage):
    answer = render_draft(draft("A claim [1, 1].", [1, 1]), [passage])
    assert answer.answer == "A claim. [1]"
    assert answer.citations == [1]


@pytest.mark.parametrize(
    "text,citations",
    [
        ("Claim", [0]),
        ("Claim", [2]),
        ("Claim [7]", [1]),
        ("In {paper:2}, a claim.", [1]),
        ("In {paper:x}, a claim.", [1]),
        ("Claim [1-3]", [1]),
        ("   ", [1]),
    ],
)
def test_draft_does_not_guess_or_silently_remove_invalid_support(passage, text, citations):
    with pytest.raises(ValueError):
        render_draft(draft(text, citations), [passage])


def test_every_paragraph_requires_sources():
    with pytest.raises(ValidationError):
        draft("Unsourced claim", [])


def test_supported_requires_paragraphs(passage):
    with pytest.raises(ValueError):
        render_draft(AnswerDraft(evidence_status="supported", paragraphs=[]), [passage])


def test_insufficient_draft_has_no_prose_or_citations(passage):
    answer = render_draft(AnswerDraft(evidence_status="insufficient", paragraphs=[]), [passage])
    assert answer.evidence_status == "insufficient"
    assert answer.citations == []
    with pytest.raises(ValueError):
        render_draft(
            AnswerDraft(evidence_status="insufficient", paragraphs=draft().paragraphs), [passage]
        )


def test_schema_limits_generation_to_actual_evidence_ids():
    items = generation_schema(3)["$defs"]["CitedParagraph"]["properties"]["citations"]["items"]
    assert items["enum"] == [1, 2, 3]


def test_multiple_paragraphs_keep_local_source_numbers(passage):
    other = passage.model_copy(update={"title": "Another paper", "id": "another"})
    response = AnswerDraft(
        evidence_status="supported",
        paragraphs=[
            CitedParagraph(text="{paper:2} describes method B.", citations=[2]),
            CitedParagraph(text="{paper:1} describes method A.", citations=[1]),
        ],
    )
    answer = render_draft(response, [passage, other])
    assert (
        answer.answer
        == "“Another paper” describes method B. [2]\n\n“Unit fixture” describes method A. [1]"
    )
    assert answer.citations == [2, 1]
