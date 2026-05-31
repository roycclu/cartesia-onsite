from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.call_state import CallState
from app.orchestration import InsuranceOrchestrator


VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
APP_STARTED_AT = datetime.now(timezone.utc)

sessions: dict[str, CallState] = {}

_orchestrator: InsuranceOrchestrator | None = None
_transcriber = None
_tts = None


def set_orchestrator(orchestrator: InsuranceOrchestrator) -> None:
    global _orchestrator
    _orchestrator = orchestrator


def get_orchestrator() -> InsuranceOrchestrator:
    if _orchestrator is None:
        raise RuntimeError("Application startup has not initialized the orchestrator.")
    return _orchestrator


def set_transcriber(transcriber) -> None:
    global _transcriber
    _transcriber = transcriber


def get_transcriber():
    if _transcriber is None:
        raise RuntimeError("Application startup has not initialized the transcriber.")
    return _transcriber


def set_tts(tts) -> None:
    global _tts
    _tts = tts


def get_tts():
    if _tts is None:
        raise RuntimeError("Application startup has not initialized TTS.")
    return _tts


def read_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip() or "unknown"
    except FileNotFoundError:
        return "unknown"
