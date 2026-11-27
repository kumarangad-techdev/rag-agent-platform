"""Turn uploaded PDF, TXT and Markdown files into text sections with location metadata."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from ..errors import IngestError

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


@dataclass(frozen=True)
class Section:
    """A piece of a document. ``page`` is set for PDFs, ``heading`` for Markdown."""

    text: str
    page: int | None = None
    heading: str | None = None


def file_type(filename: str) -> str:
    ext = PurePath(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise IngestError(f"Unsupported file type '{ext or filename}'. Allowed: {allowed}.")
    return "md" if ext in {".md", ".markdown"} else ext.lstrip(".")


def load_document(filename: str, data: bytes) -> list[Section]:
    kind = file_type(filename)
    if kind == "pdf":
        sections = _load_pdf(data)
    elif kind == "md":
        sections = _load_markdown(_decode(data))
    else:
        sections = [Section(text=_decode(data))]
    sections = [s for s in sections if s.text.strip()]
    if not sections:
        hint = " (scanned PDFs need OCR, which is not supported)" if kind == "pdf" else ""
        raise IngestError(f"'{filename}' contains no extractable text{hint}.")
    return sections


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise IngestError("Text files must be UTF-8 encoded.") from exc


def _load_pdf(data: bytes) -> list[Section]:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IngestError("Encrypted PDFs are not supported.")
        return [
            Section(text=page.extract_text() or "", page=number)
            for number, page in enumerate(reader.pages, start=1)
        ]
    except (PdfReadError, ValueError, KeyError) as exc:
        raise IngestError(f"Could not read PDF: {exc}") from exc


def _load_markdown(text: str) -> list[Section]:
    """Split Markdown at headings so each chunk can cite the section it came from."""
    sections: list[Section] = []
    heading: str | None = None
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(Section(text=body, heading=heading))
        buffer.clear()

    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2)
            buffer.append(match.group(2))  # keep the heading text searchable
        else:
            buffer.append(line)
    flush()
    return sections
