"""Extract page-aware passages and incrementally persist explicit embeddings in Chroma."""

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable

import pymupdf as fitz
from filelock import FileLock, Timeout

from config import AppError, Settings, get_settings
from models import Chunk
from storage import embed, index_signature, open_collection

EXTRACTION_VERSION = "native-order-titles-v2"


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    text = text.translate(str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}))
    text = re.sub(r"(\w)-\n(?=[a-z])", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    """Bounded windows ending at word boundaries; overlap stays within the same page."""
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    text = clean_text(text)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max(overlap + 1, size // 2), end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Move backwards to include an entire overlapping word when possible.
        boundary = text.rfind(" ", start + 1, next_start)
        if overlap and boundary >= 0 and end - boundary < size:
            next_start = boundary + 1
        start = next_start
    return [chunk for chunk in chunks if chunk]


def paper_identity(filename: str) -> str:
    return hashlib.sha256(filename.encode("utf-8")).hexdigest()[:24]


def chunk_id(paper_id: str, page: int, chunk_index: int) -> str:
    return f"{paper_id}-page-{page}-chunk-{chunk_index}"


def paper_title(document, path: Path) -> str:
    """Prefer metadata, then a prominent first-page heading; never ask an LLM to invent it."""
    raw_title = clean_text((document.metadata or {}).get("title", "") or "")
    if 4 <= len(raw_title) <= 300 and raw_title.lower() not in {
        "untitled",
        "document",
        "microsoft word",
    }:
        return raw_title
    if len(document):
        page = document[0]
        lines = []
        for block in page.get_text("dict", flags=fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_IMAGES)[
            "blocks"
        ]:
            for line in block.get("lines", []):
                # Exclude rotated arXiv stamps, running headers, and lower-page headings.
                if line.get("dir", (1, 0))[0] < 0.9:
                    continue
                if not page.rect.height * 0.05 < line["bbox"][1] < page.rect.height * 0.4:
                    continue
                text = clean_text("".join(span["text"] for span in line["spans"]))
                if len(text) >= 3 and any(char.isalpha() for char in text):
                    lines.append((max(span["size"] for span in line["spans"]), line["bbox"], text))
        if lines:
            largest = max(size for size, _, _ in lines)
            prominent = sorted(
                (item for item in lines if item[0] >= largest * 0.92),
                key=lambda item: (item[1][1], item[1][0]),
            )
            heading = []
            bottom = None
            for _, box, text in prominent:
                if bottom is not None and box[1] - bottom > largest * 1.6:
                    break
                heading.append(text)
                bottom = box[3]
            title = clean_text(" ".join(heading))
            if largest >= 12 and 15 <= len(title) <= 300:
                return title
    return re.sub(r"[_-]+", " ", path.stem)


def page_text(page) -> str:
    # Native PDF order keeps column paragraphs intact. sort=True's line reconstruction
    # interleaves left/right columns in common ACL/ICLR research PDFs.
    return clean_text(page.get_text("text", sort=False))


def extract_chunks(path: Path, settings: Settings) -> tuple[list[Chunk], int, int]:
    filename = path.relative_to(settings.papers_path).as_posix()
    paper_id = paper_identity(filename)
    chunks = []
    empty_pages = 0
    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise AppError(f"{filename}: encrypted PDF; provide an unlocked copy.")
            title = paper_title(document, path)
            page_count = len(document)
            for page_number, page in enumerate(document, start=1):
                text = page_text(page)
                if not text:
                    empty_pages += 1
                    continue
                for index, part in enumerate(
                    chunk_text(text, settings.chunk_size, settings.chunk_overlap)
                ):
                    chunks.append(
                        Chunk(
                            id=chunk_id(paper_id, page_number, index),
                            text=part,
                            paper_id=paper_id,
                            title=title,
                            filename=filename,
                            page=page_number,
                            chunk_index=index,
                        )
                    )
    except AppError:
        raise
    except Exception as exc:
        raise AppError(
            f"{filename}: could not read this PDF; check that it opens correctly."
        ) from exc
    if not chunks:
        raise AppError(f"{filename}: no usable text. Scanned PDFs require OCR before ingestion.")
    return chunks, page_count, empty_pages


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ingest(settings: Settings, report: Callable[[str], None] = print) -> dict:
    if not settings.papers_path.is_dir():
        raise AppError(f"Paper directory missing. Create {settings.papers_path} and add PDFs.")
    paths = sorted(
        p
        for p in settings.papers_path.rglob("*")
        if p.suffix.lower() == ".pdf" and p.is_file() and not p.is_symlink()
    )
    report(f"Found {len(paths)} research papers")
    if not paths:
        raise AppError(f"No PDF files found. Add research PDFs to {settings.papers_path}.")
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(str(settings.chroma_path / "ingestion.lock"), timeout=0):
            return _index(paths, settings, report)
    except Timeout as exc:
        raise AppError("Another ingestion is running. Wait for it to finish, then retry.") from exc


def _index(paths: list[Path], settings: Settings, report: Callable[[str], None]) -> dict:
    collection = open_collection(settings, create=True)
    manifest_path = settings.chroma_path / (
        hashlib.sha256(settings.collection_name.encode()).hexdigest()[:16] + "-manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
        if not isinstance(manifest, dict):
            manifest = {}
    except (ValueError, OSError):
        manifest = {}  # Deterministic upserts safely recover a missing/corrupt manifest.
    indexed = skipped = failed = 0
    for path in paths:
        filename = path.relative_to(settings.papers_path).as_posix()
        identity = paper_identity(filename)
        report(f"\nProcessing: {filename}")
        try:
            digest = file_hash(path)
            previous = manifest.get(filename, {})
            stored = collection.get(where={"paper_id": identity}, include=[])["ids"]
            if (
                previous.get("hash") == digest
                and previous.get("signature") == index_signature(settings)
                and previous.get("extraction_version") == EXTRACTION_VERSION
                and len(stored) == previous.get("chunks")
                and stored
            ):
                report("Unchanged; using persisted chunks.")
                skipped += 1
                continue
            chunks, pages, empty_pages = extract_chunks(path, settings)
            report(f"Pages extracted: {pages}; pages without text: {empty_pages}")
            report(f"Chunks created: {len(chunks)}")
            # Encode before altering the index so embedding failure preserves prior data.
            vectors = embed([chunk.text for chunk in chunks], settings)
            for offset in range(0, len(chunks), settings.batch_size):
                batch = chunks[offset : offset + settings.batch_size]
                collection.upsert(
                    ids=[chunk.id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata() for chunk in batch],
                    embeddings=vectors[offset : offset + settings.batch_size],
                )
            obsolete = sorted(set(stored) - {chunk.id for chunk in chunks})
            for offset in range(0, len(obsolete), settings.batch_size):
                collection.delete(ids=obsolete[offset : offset + settings.batch_size])
            manifest[filename] = {
                "hash": digest,
                "signature": index_signature(settings),
                "extraction_version": EXTRACTION_VERSION,
                "chunks": len(chunks),
            }
            temporary = manifest_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary.replace(manifest_path)
            indexed += 1
        except (AppError, OSError) as exc:
            failed += 1
            report(f"Skipped: {exc}")
    total = collection.count()
    report(f"\nIndexed {indexed} papers; unchanged {skipped}; failed {failed}.")
    report(f"Vector database contains {total} chunks.\nVector database: {settings.chroma_path}")
    return {"indexed": indexed, "skipped": skipped, "failed": failed, "chunks": total}


def main() -> int:
    try:
        result = ingest(get_settings())
        return 1 if result["failed"] else 0
    except AppError as exc:
        print(f"Ingestion: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print(
            "Ingestion failed. Check disk space and database access, then retry. "
            "Run with a fresh CHROMA_PATH if the database is damaged.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
