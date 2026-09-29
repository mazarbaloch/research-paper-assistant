"""Local Gradio UI. Importing or starting this module never builds the index."""

import html
import logging
import re
import shutil
from pathlib import Path

import gradio as gr
import pymupdf as fitz

from ai_service import generate_answer
from config import AppError, Settings, get_settings
from ingest import file_hash, ingest
from models import GroundedAnswer, Passage
from retrieval import retrieve
from storage import index_stats

LOGGER = logging.getLogger(__name__)
EXAMPLES = [
    "What limitations of RAG are discussed?",
    "How is retrieval evaluated?",
    "Which papers discuss reranking?",
    "Compare the retrieval methods in these papers.",
    "What problems do the papers identify with basic vector search?",
]
CSS = """
.gradio-container { max-width: 1180px !important; margin: auto; }
#hero { padding: 26px 0 8px; }
#hero h1 { letter-spacing: -0.035em; font-size: 34px; }
#hero p { max-width: 800px; color: #64748b; }
.source-card { border: 1px solid #94a3b866; border-radius: 10px;
  padding: 14px 16px; margin-top: 12px; }
.source-card summary { cursor: pointer; font-weight: 600; }
.source-card .provenance { color: #64748b; font-size: 12px; margin: 8px 0; }
.source-card blockquote { margin: 12px 0 0; padding-left: 12px;
  border-left: 3px solid #10b981; white-space: pre-wrap; }
.answer-text { white-space: pre-wrap; line-height: 1.7; }
.evidence-label { color: #64748b; font-size: 12px; margin-bottom: 10px; }
"""


def render_answer(answer: GroundedAnswer, passages: list[Passage]) -> str:
    label = (
        "Supported by retrieved passages"
        if answer.evidence_status == "supported"
        else "Insufficient evidence"
    )
    content = (
        f'<div class="evidence-label">{label}</div>'
        f'<div class="answer-text">{html.escape(answer.answer)}</div>'
    )
    for number in sorted(answer.citations):
        passage = passages[number - 1]
        content += (
            '<details class="source-card" open>'
            f"<summary>[{number}] {html.escape(passage.title)} — Page {passage.page}</summary>"
            f'<div class="provenance">{html.escape(passage.filename)} · '
            f"Cosine distance: {passage.distance:.3f}</div>"
            f"<blockquote>{html.escape(passage.text)}</blockquote></details>"
        )
    return content


def source_files(passages: list[Passage], citations: list[int], settings: Settings) -> list[str]:
    files = []
    root = settings.papers_path.resolve()
    for number in citations:
        path = (root / passages[number - 1].filename).resolve()
        if path.is_relative_to(root) and path.suffix.lower() == ".pdf" and path.is_file():
            if str(path) not in files:
                files.append(str(path))
    return files


def retrieval_question(question: str, history: list[dict[str, str]]) -> str:
    # Only referential follow-ups need previous questions. Unrelated standalone questions
    # must not inherit old topics (e.g. "How is retrieval evaluated?" after hallucinations).
    if not re.search(
        r"\b(those|that|them|their|it|this (?:method|approach|finding|result)|"
        r"these (?:methods|approaches|findings|limitations|results))\b|^(?:what|how) about\b",
        question,
        re.IGNORECASE,
    ):
        return question
    previous = [item["content"] for item in history if item["role"] == "user"][-2:]
    return "\n".join([question] + [f"Previous question: {text[:500]}" for text in previous])


def ask(question: str, chat: list, history: list, top_k: int, settings: Settings):
    chat, history = list(chat or []), list(history or [])
    question = question.strip()
    if not question:
        return "", chat, history, []
    chat.append({"role": "user", "content": question})
    try:
        if len(question) > 4000:
            raise AppError("Please shorten your question to 4,000 characters or fewer.")
        query = retrieval_question(question, history)
        passages = retrieve(query, settings, int(top_k))
        context = history if query != question else []
        answer = generate_answer(question, passages, settings, context)
        chat.append({"role": "assistant", "content": render_answer(answer, passages)})
        # Keep the visible conversation, but start a new context chain for a new topic.
        # Otherwise a later "that explanation" can revive older unrelated questions.
        history = list(context)
        history.extend(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer.answer}]
        )
        history = history[-settings.history_turns * 2 :] if settings.history_turns else []
        return "", chat, history, source_files(passages, answer.citations, settings)
    except AppError as exc:
        chat.append({"role": "assistant", "content": html.escape(str(exc))})
    except Exception:
        LOGGER.exception("Question failed")
        chat.append(
            {
                "role": "assistant",
                "content": (
                    "Could not complete the question. Check that the index is readable and "
                    "Ollama is running, then retry. See the terminal for technical details."
                ),
            }
        )
    return "", chat, history, []


def status_text(settings: Settings) -> str:
    try:
        papers, chunks = index_stats(settings)
        return f"**{papers}** papers indexed · **{chunks:,}** passages · Local inference"
    except AppError as exc:
        return html.escape(str(exc))
    except Exception:
        LOGGER.exception("Status failed")
        return "Could not read the index. Check database permissions and re-run ingestion."


def add_and_index(files: list[str] | None, settings: Settings) -> str:
    """Copy validated uploads into the library; never overwrite an existing different PDF."""
    messages = []
    try:
        settings.papers_path.mkdir(parents=True, exist_ok=True)
        for source in files or []:
            path = Path(source)
            if path.suffix.lower() != ".pdf":
                messages.append(f"Skipped {path.name}: upload a PDF.")
                continue
            try:
                with fitz.open(path) as document:
                    if not document.is_pdf or document.needs_pass:
                        messages.append(f"Skipped {path.name}: provide a readable, unlocked PDF.")
                        continue
                name = re.sub(r"[^\w. -]", "_", path.name)
                target = settings.papers_path / name
                if target.exists() and file_hash(target) != file_hash(path):
                    target = target.with_name(f"{target.stem}-{file_hash(path)[:10]}.pdf")
                if not target.exists():
                    shutil.copyfile(path, target)
            except (OSError, fitz.FileDataError):
                messages.append(f"Skipped {path.name}: the file could not be read as a PDF.")
        ingest(settings, report=messages.append)
    except AppError as exc:
        messages.append(str(exc))
    except Exception:
        LOGGER.exception("Ingestion failed")
        messages.append("Indexing failed. Check disk space and directory permissions, then retry.")
    return "\n".join(messages)


def build_app(settings: Settings | None = None) -> gr.Blocks:
    settings = settings or get_settings()
    with gr.Blocks(title="Research Paper Knowledge Assistant", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Research Paper Knowledge Assistant\n"
            "Ask questions across a collection of research papers. Answers are grounded in "
            "retrieved evidence from the indexed PDFs.",
            elem_id="hero",
        )
        status = gr.Markdown("Loading library status…")
        with gr.Tab("Ask your papers"):
            with gr.Row():
                with gr.Column(scale=4):
                    chat = gr.Chatbot(
                        label="Research conversation",
                        height=530,
                        layout="panel",
                        sanitize_html=True,
                        render_markdown=True,
                        placeholder="Add papers in the Library tab, then ask a research question.",
                    )
                    question = gr.Textbox(
                        label="Your question",
                        lines=2,
                        max_lines=6,
                        placeholder="What do these papers say about…?",
                    )
                    with gr.Row():
                        submit = gr.Button("Ask papers", variant="primary")
                        clear = gr.Button("New conversation")
                    gr.Examples(examples=[[item] for item in EXAMPLES], inputs=question)
                with gr.Column(scale=1, min_width=230):
                    gr.Markdown(
                        "### Evidence first\nEvery answer uses a fresh search. "
                        "Expand source cards under an answer to inspect its passages."
                    )
                    top_k = gr.Slider(
                        1, 20, value=settings.top_k, step=1, label="Passages to retrieve"
                    )
                    downloads = gr.File(
                        label="Original PDFs · latest answer",
                        file_count="multiple",
                        interactive=False,
                    )
                    gr.Markdown(
                        "Page numbers refer to PDF pages. Retrieval distance describes "
                        "similarity, not whether a claim is supported."
                    )
                    gr.Markdown(
                        f"**Generation**\n{html.escape(settings.generation_model)}\n\n"
                        f"**Embeddings**\n{html.escape(settings.embedding_model)}"
                    )
        with gr.Tab("Library"):
            gr.Markdown(
                "### Build your knowledge base\nUpload legally obtained research PDFs, "
                "or place them in the paper directory. Indexing skips unchanged files. "
                "Scanned papers need OCR first."
            )
            uploads = gr.File(
                label="Research PDFs", file_count="multiple", file_types=[".pdf"], type="filepath"
            )
            index_button = gr.Button("Add PDFs and index library", variant="primary")
            log = gr.Textbox(label="Indexing report", lines=10, interactive=False)
            gr.Markdown(
                f"Paper directory: `{settings.papers_path}`\n\n"
                "You can also run `uv run python ingest.py` in the project directory. "
                "Avoid running CLI ingestion while answering questions."
            )
        history = gr.State([])
        args = [question, chat, history, top_k]
        outputs = [question, chat, history, downloads]
        # Serialize reads and writes from this UI so a question sees a completed ingestion.
        for event in (submit.click, question.submit):
            event(
                lambda q, c, h, k: ask(q, c, h, k, settings),
                args,
                outputs,
                concurrency_id="library",
                concurrency_limit=1,
            )
        clear.click(
            lambda: ("", [], [], []), outputs=outputs, concurrency_id="library", concurrency_limit=1
        )
        chat.clear(
            lambda: ([], []),
            outputs=[history, downloads],
            concurrency_id="library",
            concurrency_limit=1,
        )
        index_button.click(
            lambda files: add_and_index(files, settings),
            uploads,
            log,
            concurrency_id="library",
            concurrency_limit=1,
        ).then(lambda: status_text(settings), outputs=status)
        demo.load(lambda: status_text(settings), outputs=status)
    return demo


if __name__ == "__main__":
    try:
        config = get_settings()
        build_app(config).queue().launch(
            server_name="127.0.0.1",
            server_port=config.server_port,
            share=False,
            theme=gr.themes.Soft(
                primary_hue="emerald",
                neutral_hue="slate",
                font=["system-ui", "sans-serif"],
                font_mono=["Consolas", "monospace"],
            ),
            css=CSS,
            max_file_size="100mb",
        )
    except AppError as exc:
        raise SystemExit(str(exc)) from exc
    except OSError as exc:
        raise SystemExit(
            "Could not start the local server. Set GRADIO_SERVER_PORT to an unused port "
            "(for example 7861) and check local network permissions."
        ) from exc
