from __future__ import annotations

import pytest

from rag_agent_platform.errors import IngestError
from rag_agent_platform.rag.chunking import chunk_text
from rag_agent_platform.rag.loaders import load_document

from .conftest import ROOT


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", 200, 20) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ", 200, 20) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} has some words in it." for i in range(80))
    chunks = chunk_text(text, 300, 60)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # every chunk after the first starts with words from the end of the previous chunk
    for previous, current in zip(chunks, chunks[1:], strict=False):
        first_word = current.split()[0]
        assert first_word in previous[-80:]


def test_no_text_is_lost():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), 250, 0)
    assert " ".join(chunks).split() == words


def test_giant_word_is_split():
    chunks = chunk_text("x" * 1000, 300, 0)
    assert all(len(c) <= 300 for c in chunks)
    assert "".join(chunks) == "x" * 1000


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", 100, 100)


def test_markdown_sections_keep_headings():
    md = (
        b"# Title\nIntro text.\n\n## Leave\nTwenty days.\n"
        b"```\n# not a heading\n```\n## Pay\nMonthly."
    )
    sections = load_document("notes.md", md)
    assert [s.heading for s in sections] == ["Title", "Leave", "Pay"]
    assert "# not a heading" in sections[1].text


def test_pdf_pages_are_numbered():
    data = (ROOT / "sample_docs" / "rag_primer.pdf").read_bytes()
    sections = load_document("rag_primer.pdf", data)
    assert [s.page for s in sections] == [1, 2, 3]
    assert "Retrieval-Augmented Generation" in sections[0].text


def test_unsupported_type_is_rejected():
    with pytest.raises(IngestError, match="Unsupported"):
        load_document("photo.png", b"\x89PNG")


def test_empty_and_non_utf8_files_are_rejected():
    with pytest.raises(IngestError, match="no extractable text"):
        load_document("empty.txt", b"   ")
    with pytest.raises(IngestError, match="UTF-8"):
        load_document("latin.txt", "café".encode("latin-1"))


def test_corrupt_pdf_is_rejected():
    with pytest.raises(IngestError):
        load_document("broken.pdf", b"%PDF-1.4 not really a pdf")
