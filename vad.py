from __future__ import annotations

import audioop
from dataclasses import dataclass, field

import numpy as np


try:
    from silero_vad import VADIterator, load_silero_vad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    VADIterator = None
    load_silero_vad = None


@dataclass
class VADResult:
    speech_detected: bool
    silence_frames: int
    end_of_turn: bool


@dataclass
class TurnDetector:
    sample_rate: int = 8000
    silence_threshold: int = 250
    silence_frames_for_turn: int = 8
    use_silero: bool = True
    silence_frames: int = 0
    _iterator: object | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.use_silero and load_silero_vad and VADIterator:
            model = load_silero_vad()
            self._iterator = VADIterator(model, sampling_rate=self.sample_rate)

    def reset(self) -> None:
        self.silence_frames = 0
        if self._iterator and hasattr(self._iterator, "reset_states"):
            self._iterator.reset_states()

    def ingest(self, chunk: bytes) -> VADResult:
        speech_detected = self._detect_speech(chunk)
        if speech_detected:
            self.silence_frames = 0
        else:
            self.silence_frames += 1
        return VADResult(
            speech_detected=speech_detected,
            silence_frames=self.silence_frames,
            end_of_turn=self.silence_frames >= self.silence_frames_for_turn,
        )

    def _detect_speech(self, chunk: bytes) -> bool:
        if self._iterator is not None:
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            event = self._iterator(pcm, return_seconds=False)
            return bool(event)
        return audioop.rms(chunk, 2) >= self.silence_threshold
