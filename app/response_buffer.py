from __future__ import annotations


class ResponseBuffer:
    def __init__(self) -> None:
        self.active = False
        self.superseded = False
        self.sentences_sent = 0

    def start(self) -> None:
        self.active = True
        self.superseded = False
        self.sentences_sent = 0

    def supersede(self) -> None:
        self.superseded = True
        self.active = False

    def sentence_complete(self) -> None:
        self.sentences_sent += 1
