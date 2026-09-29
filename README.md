# Research Paper Knowledge Assistant

A standalone Python application for asking questions across real research-paper PDFs.
PyMuPDF extracts page-aware passages, Sentence Transformers embeds them, persistent Chroma
retrieves evidence, and a local Ollama model produces a validated answer with inspectable sources.
No cloud generation, cloud vector database, LangChain, or LlamaIndex is used.

## What it does

- Builds a searchable semantic knowledge base from your PDFs.
- Answers using fresh retrieved passages, with paper title, PDF page, filename, and exact text.
- Rejects invalid citation numbers and mismatches between inline citations and source records.
- Abstains when the model reports insufficient evidence, displaying no misleading sources.
- Supports conversational follow-ups, library uploads, and original-PDF downloads.
- Keeps the index across restarts and skips unchanged papers on subsequent ingestion runs.

## Architecture

```text
Research PDFs
     ↓
PDF extraction (PyMuPDF, one page at a time)
     ↓
Page-aware chunking (1,000 characters, ~150-character overlap)
     ↓
SentenceTransformer embeddings (explicit, normalized)
     ↓
Persistent Chroma database (cosine distance)
     ↓
Question embedding
     ↓
Semantic retrieval (top 5 by default)
     ↓
Retrieved evidence
     ↓
Ollama (qwen3.5:9b)
     ↓
Pydantic validation + citation checks
     ↓
Grounded answer + citations + source passages
```

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and
[Ollama](https://ollama.com/download). Use Python 3.11–3.13; `uv` can install Python if needed.
From this project directory:

```bash
uv sync
ollama pull qwen3.5:9b
```

Start the Ollama desktop application, or run `ollama serve` if it is not already running.
The default model requires several gigabytes of disk and substantial available RAM/VRAM;
CPU inference works but can be slow. A smaller **local** model can be selected with
`GENERATION_MODEL` if necessary.

The Sentence Transformers model `sentence-transformers/all-MiniLM-L6-v2` downloads from
Hugging Face on first ingestion/query if not cached. Dependency/model downloads require
internet access; inference, embeddings, retrieval, and PDF processing run locally afterward.
For a fully offline session after caching the embedding model, set `HF_HUB_OFFLINE=1`.
The UI binds to `127.0.0.1` and does not create a public sharing link. Ollama must also be local.

## Add papers

Put legally obtained research PDFs in:

```text
data/papers/
```

Subdirectories and uppercase `.PDF` extensions are supported. Alternatively, use the Library
tab to upload PDFs (100 MB per file maximum) and index them. Uploading a different PDF with an
existing name creates a hash-suffixed filename instead of overwriting the original.
Papers and the vector database are excluded from Git. No fabricated papers or sample corpus
are bundled. Scanned PDFs need OCR before adding them.

## Build the knowledge base

```bash
uv run python ingest.py
```

The command reports discovered papers, extracted pages, empty pages, created chunks, unchanged
files, and failures. An unreadable/encrypted/textless PDF is reported without stopping other
papers. A nonzero exit code indicates at least one failure or a setup error.

PDFs are hashed in streaming blocks. A manifest records successful content hashes and index
configuration. Deterministic IDs combine relative filename identity, PDF page, and within-page
chunk position. Re-running ingestion skips unchanged papers when their stored chunk count
matches. Changed files are upserted and obsolete chunks removed. Manifest writes are atomic;
interrupted indexing can be retried safely without accumulating duplicate chunk IDs.
An ingestion lock prevents two ingestion commands from writing concurrently.

Removing or renaming a PDF **does not delete its old indexed passages**. Keep original PDFs
in place to preserve downloads. To rebuild a clean library, select a new `COLLECTION_NAME`
(or a fresh `CHROMA_PATH`) and index the desired PDFs again. This also applies after changing
embedding or chunking settings: the app refuses to mix incompatible index configurations.
Do not run CLI ingestion while the UI is answering questions; multi-batch updates are not a
transaction. Ingestion and questions launched from the UI are serialized.

## Run the application

```bash
uv run python app.py
```

Open [the local application](http://127.0.0.1:7860). Startup reads index status only; it never
extracts PDFs, downloads embedding weights, or rebuilds the index. With an empty library, the
UI still starts and explains how to add papers. Use **New conversation** to clear context.

Each supported answer contains its own numbered source cards. Only cited passages are shown;
numbers retain their original retrieval positions (e.g. `[2]` and `[5]` are not renumbered).
Original PDFs for the latest answer appear in the download panel when still available.

## How the system works

**Extraction and chunks.** PyMuPDF extracts native PDF text order page by page. This avoids
interleaving the left and right columns in common research-paper layouts. Whitespace, ligatures,
soft hyphens, and obvious line-break hyphenation are cleaned. Short paragraphs are combined into readable
windows of up to 1,000 characters, with roughly 150 characters of word-aware overlap. Chunks
never cross pages. Titles come from useful PDF metadata, or the prominent heading on the first
page (ignoring rotated arXiv stamps and running headers), with a readable filename as fallback.
An extraction-version marker automatically refreshes older indexed chunks on the next ingestion
when extraction changes; unchanged current-version papers still skip processing.
Pages refer to one-based positions in the PDF, not its printed page labels.

**Embeddings and storage.** Both ingestion and queries explicitly call
`SentenceTransformer.encode(..., normalize_embeddings=True)`. Chroma receives these vectors
directly, along with chunk text and provenance; its implicit embedding function is disabled.
`chromadb.PersistentClient(path=...)` stores the cosine-distance collection on disk.

**Retrieval and follow-ups.** Each request embeds the current question, then retrieves the
nearest passages (default five). Referential follow-ups such as "Which of those?" include up
to two recent user questions as context. Standalone questions do not inherit unrelated old
topics. There is no query-rewriting LLM. Previous assistant answers are excluded from query
embeddings. For follow-ups, bounded conversation is passed separately to generation, labeled
as context, never evidence. Each standalone question starts a new hidden context chain while
older messages remain visible, preventing subsequent follow-ups from reviving unrelated topics.
Every follow-up uses a fresh search.

**Grounding.** Only retrieved research passages, the question, and bounded conversation context
are supplied to Ollama. The prompt explicitly forbids external knowledge, invented findings,
invented citations, and instructions embedded in PDFs. The model returns `evidence_status`
and a list of paragraphs, each containing `text` and supporting `citations`, through
[Ollama's JSON schema output](https://docs.ollama.com/capabilities/structured-outputs).
Pydantic rejects malformed responses and unexpected fields. The prompt asks for direct,
natural answers, naming papers rather than starting with "The provided evidence indicates".

**Citations.** The model assigns sources once per paragraph. The application generates both
inline markers such as `[1] [3]` and the final answer's citation array from those assignments;
the model no longer has to keep two redundant citation representations in sync. The generation
schema restricts source numbers to the actual retrieved evidence, and server-side validation
still rejects invalid numbers. Every supported paragraph requires sources. Redundant valid
inline citations are normalized, while invalid ones are rejected. To name a paper naturally,
the model uses `{paper:1}`; the application substitutes the exact title belonging to that
paragraph's cited source. Authors are not guessed. The public `GroundedAnswer` remains
`answer`, `evidence_status`, and `citations`.

A malformed response gets one targeted retry, then an actionable error; the invalid answer
is never shown. Validation reasons are logged locally for diagnosis. Source titles and
passage text come from the PDFs and stored metadata, not model-generated references.
PDF text and model output are HTML-escaped before display.

**Insufficient evidence.** An empty evidence list abstains without calling the model. When
Ollama marks evidence insufficient, the application returns a fixed, cautious abstention and
no citations. Semantic similarity alone is never treated as proof of a claim. A lack of support
in the retrieved passages is not proof that the entire library lacks an answer.

## Configuration

Environment variables are read by `config.py`. Paths are relative to the project directory
unless absolute. The application does not automatically load `.env` files.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama endpoint |
| `GENERATION_MODEL` | `qwen3.5:9b` | Installed local generation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `TOP_K` | `5` | Retrieved passages, 1–20; adjustable in UI |
| `CHROMA_PATH` | `data/chroma` | Persistent database and manifests |
| `PAPERS_PATH` | `data/papers` | PDF library |
| `COLLECTION_NAME` | `research_papers` | Chroma collection |
| `CHUNK_SIZE` | `1000` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `150` | Approximate overlapping characters |
| `BATCH_SIZE` | `64` | Embedding and database batch size |
| `HISTORY_TURNS` | `3` | Recent conversation pairs sent to generation |
| `OLLAMA_TIMEOUT` | `180` | Generation request timeout, seconds |
| `GRADIO_SERVER_PORT` | `7860` | Local UI port |

PowerShell example:

```powershell
$env:TOP_K = "8"
$env:OLLAMA_TIMEOUT = "300"
uv run python app.py
```

Bash example:

```bash
TOP_K=8 OLLAMA_TIMEOUT=300 uv run python app.py
```

## Project structure

```text
app.py                 Gradio conversation, source cards, uploads, status
config.py              Environment configuration and actionable errors
ingest.py              PDF extraction, chunking, incremental indexing CLI
storage.py             Persistent Chroma access and normalized embeddings
retrieval.py           Query embeddings and semantic search records
ai_service.py          Evidence prompt, Ollama generation, citation validation
models.py              Pydantic chunk, passage, and answer schemas
pyproject.toml          Runtime and test dependencies
uv.lock                Resolved reproducible dependency versions
data/papers/README.md  Instructions for adding real PDFs
data/chroma/           Generated local database (ignored by Git)
tests/                 Focused deterministic regression tests
```

## Verification

```bash
uv sync --locked
uv pip check
uv run ruff check .
uv run python -m compileall -q app.py config.py ingest.py storage.py retrieval.py ai_service.py models.py
uv run pytest -q
```

Tests cover chunk bounds/overlap, empty pages, deterministic IDs, incremental indexing,
changed-file cleanup, actual temporary persistent Chroma queries/upserts, missing-index errors,
configuration mismatch, output parsing/retry, invalid citations, abstention, safe source display,
follow-up context, and startup without indexing. Unit tests use isolated fixtures/stubs at model
and extraction boundaries; they do not require Ollama, download model weights, or fabricate a
research corpus. Real end-to-end research ingestion requires your PDFs.

On September 29, 2026, local verification reindexed 11 user-provided PDFs into 1,173 passages
with corrected reading order and extracted titles. A second ingestion skipped all 11 files
and retained exactly 1,173 passages. The real MiniLM
model produced normalized 384-dimensional embeddings, and the installed `qwen3.5:9b` returned
valid schema output. The unsupported GPT-8 / 99% accuracy question returned `insufficient`
with no citations against the real library. These local PDFs and generated data remain ignored
by Git and are not bundled with the application.

The running Gradio endpoint also returned a supported answer with source cards and downloadable
original PDFs, then handled a conversational follow-up with fresh citations. All five passages
retrieved for the dense-retrieval question were checked against their original PDF page text.
Live checks covered "How is retrieval evaluated?", RAG limitations, reranking, evaluation metrics,
and the unsupported GPT-8 question. Answers now name papers naturally. The formatting fix keeps
the existing MiniLM and Qwen defaults: the observed failures involved redundant citation
formatting, missing titles, PDF reading order, and unrelated conversation context rather than
an embedding-model compatibility failure.

The deterministic suite passed 60 tests; dependency consistency, imports, syntax, and Ruff checks
passed. Chroma emitted one upstream deprecation warning during its configuration inspection.

## Troubleshooting

- **Ollama unavailable:** start the Ollama app or `ollama serve`; confirm `ollama list` works.
- **Model missing:** run `ollama pull qwen3.5:9b` (or your configured model name).
- **Timeout:** wait for loading, increase `OLLAMA_TIMEOUT`, or use a smaller local model.
- **No papers / missing database:** add PDFs and run `uv run python ingest.py`.
- **Embedding failure:** check internet access on first use, the model name, and free memory.
- **PDF has no text:** run OCR separately and add the resulting searchable PDF.
- **Invalid response/citations:** retry and inspect the validation reason in the terminal.
  The application never fills in missing evidence or displays invalid source numbers.
- **Index configuration mismatch:** restore the original settings or choose a new collection.
- **Port already in use:** set a different `GRADIO_SERVER_PORT`.

## Limitations

- No built-in OCR; native reading order depends on the PDF producer. Complex column layouts,
  equations, tables, headers, and references can still extract
  imperfectly. Short pages are kept as short passages rather than merged across page boundaries.
- Character-based windows are approximate context units. The embedding model has a token limit;
  unusually dense/non-English text can be truncated even inside a bounded character chunk.
- Basic dense retrieval has no reranking or lexical search and can miss relevant passages.
  Nearest neighbors are not necessarily sufficient evidence. Top-k retrieval cannot establish
  exhaustive coverage or which finding is most frequent across the whole library.
- Citation validation guarantees number-to-passage integrity, **not semantic entailment**.
  Local models can still misinterpret evidence or fail to abstain. Inspect the displayed passages
  before relying on research conclusions. Prompt-injection resistance is not a formal guarantee.
- Conversation context is bounded, and including previous questions can dilute retrieval.
- The app is intended for one local user. Index updates are not fully transactional across all
  chunks; avoid external writes during queries. It is not an authenticated public hosting service.
- No PDFs are bundled; a fresh checkout requires your own legally obtained research papers.
