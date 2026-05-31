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
