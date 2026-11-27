"""Recursive character chunking with overlap: paragraphs, then sentences, then words."""

from __future__ import annotations

import re

_PARAGRAPHS = re.compile(r"\n\s*\n")
_SENTENCES = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Paragraphs are kept whole where possible; long paragraphs fall back to sentences and then
    words. Each chunk after the first starts with up to ``overlap`` characters from the end of
    the previous chunk (cut at a word boundary), so facts that straddle a boundary stay
    retrievable.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text.strip():
        return []

    chunks: list[str] = []
    current = ""
    for piece, starts_paragraph in _units(text, chunk_size):
        sep = "\n\n" if starts_paragraph else " "
        candidate = f"{current}{sep}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = _tail(current, overlap)
            fits = tail and len(tail) + 1 + len(piece) <= chunk_size
            current = f"{tail} {piece}" if fits else piece
        else:
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _units(text: str, limit: int) -> list[tuple[str, bool]]:
    """Yield (unit, starts_new_paragraph) pairs, each unit no longer than ``limit``."""
    units: list[tuple[str, bool]] = []
    for paragraph in _PARAGRAPHS.split(text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if len(paragraph) <= limit:
            units.append((paragraph, True))
            continue
        first = True
        for sentence in _SENTENCES.split(paragraph):
            parts = [sentence] if len(sentence) <= limit else _split_words(sentence, limit)
            for part in parts:
                units.append((part, first))
                first = False
    return units


def _split_words(sentence: str, limit: int) -> list[str]:
    out: list[str] = []
    current = ""
    for word in sentence.split():
        while len(word) > limit:  # a single enormous "word" such as a URL or base64 blob
            if current:
                out.append(current)
                current = ""
            out.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) <= limit:
            current = candidate
        else:
            out.append(current)
            current = word
    if current:
        out.append(current)
    return out


def _tail(text: str, size: int) -> str:
    if size <= 0:
        return ""
    tail = text[-size:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) - 1 else tail
