from __future__ import annotations

import re
from typing import Awaitable, Callable


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    matches = list(re.finditer(r"[^.!?]*[.!?]", buffer))
    if not matches:
        return [], buffer
    sentences = [match.group(0).strip() for match in matches]
    remainder = buffer[matches[-1].end() :]
    return [sentence for sentence in sentences if sentence], remainder


async def emit_sentences(text: str, sentence_handler: Callable[[str], Awaitable[None]]) -> None:
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if sentence:
            await sentence_handler(sentence)


def split_speakable_chunks(buffer: str, *, min_chars: int = 24, max_chars: int = 72) -> tuple[list[str], str]:
    chunks: list[str] = []
    remainder = buffer
    while remainder:
        boundary = _find_chunk_boundary(remainder, min_chars=min_chars, max_chars=max_chars)
        if boundary is None:
            break
        chunk = remainder[:boundary]
        remainder = remainder[boundary:]
        if chunk:
            chunks.append(chunk)
    return chunks, remainder


def _find_chunk_boundary(buffer: str, *, min_chars: int, max_chars: int) -> int | None:
    punctuation_boundary: int | None = None
    for index, char in enumerate(buffer):
        if char not in ".!?,;:":
            continue
        boundary = index + 1
        while boundary < len(buffer) and buffer[boundary].isspace():
            boundary += 1
        if boundary >= min_chars:
            punctuation_boundary = boundary
            break
    if punctuation_boundary is not None:
        return punctuation_boundary
    if len(buffer) < max_chars:
        return None
    boundary = buffer.rfind(" ", min_chars, max_chars + 1)
    if boundary == -1:
        return max_chars
    while boundary < len(buffer) and buffer[boundary].isspace():
        boundary += 1
    return boundary
