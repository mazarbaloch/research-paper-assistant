"""Shared records and strict structured generation output."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

INSUFFICIENT_ANSWER = (
    "The retrieved passages from the indexed research papers do not contain enough "
    "evidence to answer this question. Try a more specific question or retrieve more passages."
)


class Chunk(BaseModel):
    id: str
    text: str
    paper_id: str
    title: str
    filename: str
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)

    def metadata(self) -> dict:
        return self.model_dump(exclude={"id", "text"})


class Passage(BaseModel):
    id: str
    text: str
    title: str
    filename: str
    page: int = Field(ge=1)
    distance: float


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(
        min_length=1,
        max_length=16000,
        description="Answer text. For supported answers, include inline citations like [1] "
        "after every factual claim. The citations array alone is not enough.",
    )
    evidence_status: Literal["supported", "insufficient"]
    citations: list[int] = Field(
        default_factory=list,
        max_length=20,
        description="Distinct evidence numbers used inline in answer, or [] if insufficient.",
    )


class CitedParagraph(BaseModel):
    """The model assigns sources once; the application renders inline citation numbers."""

    model_config = ConfigDict(extra="forbid", strict=True)
    text: str = Field(
        min_length=1,
        max_length=3000,
        description="One concise paragraph. Use {paper:1} to name evidence 1's paper. "
        "Do not write numeric citations; the application adds them.",
    )
    citations: list[int] = Field(
        min_length=1,
        max_length=20,
        description="Evidence numbers supporting all claims in this paragraph.",
    )


class AnswerDraft(BaseModel):
    """Small generation schema, separate from the final display record."""

    model_config = ConfigDict(extra="forbid", strict=True)
    evidence_status: Literal["supported", "insufficient"]
    paragraphs: list[CitedParagraph] = Field(
        max_length=8,
        description="Supported answer paragraphs, or [] when evidence is insufficient.",
    )


def insufficient_answer() -> GroundedAnswer:
    return GroundedAnswer(answer=INSUFFICIENT_ANSWER, evidence_status="insufficient")
