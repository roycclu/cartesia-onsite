from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app import config
from app.app_state import APP_STARTED_AT, get_orchestrator, read_version, sessions
from app.call_state_manager import call_state_manager
from app.compliance import get_session_log, log_event
from app.prompts import PROMPT_VERSION
from app.audio import fail_safe_handoff, send_agent_response
from app.turn_handler import process_transcript
from mock_data.db import database_status

logger = logging.getLogger("voice_agent")

router = APIRouter()


class TextTurnRequest(BaseModel):
    session_id: str | None = None
    transcript: str


@router.get("/health")
async def health() -> dict[str, str | float]:
    return {
        "status": "ok",
        "version": read_version(),
        "database": await database_status(),
        "uptime": round((datetime.now(timezone.utc) - APP_STARTED_AT).total_seconds(), 3),
    }


@router.post("/calls/start")
async def start_call(payload: dict[str, Any]) -> dict[str, str]:
    session_id = payload.get("session_id") or str(uuid.uuid4())
    state = call_state_manager.create(session_id, payload.get("call_sid"))
    state.prompt_version = PROMPT_VERSION
    sessions[session_id] = state
    await log_event(session_id, "call_started", payload)
    return {"session_id": session_id}


@router.post("/demo/text-turn")
async def demo_text_turn(request: TextTurnRequest) -> dict[str, Any]:
    session_id = request.session_id or str(uuid.uuid4())
    session = sessions.get(session_id) or call_state_manager.get(session_id) or call_state_manager.create(session_id, None)
    session.prompt_version = PROMPT_VERSION
    sessions[session_id] = session
    result = await process_transcript(session, request.transcript)
    response_text = result.get("response_text") or result.get("response") or ""
    return {
        "session_id": session_id,
        "response_text": response_text,
        "verified": session.verified,
        "should_handoff": session.should_handoff,
        "handoff_reason": session.handoff_reason,
    }


@router.get("/sessions/{session_id}/compliance-log")
async def session_log(session_id: str) -> list[dict[str, Any]]:
    return await get_session_log(session_id)


@router.websocket("/ws/cartesia/{session_id}")
async def cartesia_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = sessions.get(session_id) or call_state_manager.get(session_id) or call_state_manager.create(session_id, None)
    session.prompt_version = PROMPT_VERSION
    sessions[session_id] = session
    try:
        while True:
            message = await websocket.receive()
            if "text" in message and message["text"] is not None:
                event = json.loads(message["text"])
                await handle_websocket_event(websocket, session, event)
            elif "bytes" in message and message["bytes"] is not None:
                logger.info("cartesia_raw_audio_ignored session_id=%s", session_id)
    except WebSocketDisconnect:
        await log_event(session_id, "call_disconnected", {"session_id": session_id})
    except Exception as exc:
        await fail_safe_handoff(websocket, session, f"unhandled_exception:{exc}")


async def handle_websocket_event(websocket: WebSocket, session, event: dict[str, Any]) -> None:
    event_type = event.get("event")
    if event_type == "start":
        session.stream_sid = event.get("stream_id") or str(uuid.uuid4())
        await log_event(session.session_id, "stream_start", event)
        await websocket.send_json({"event": "ack", "stream_id": session.stream_sid, "config": event.get("config", {})})
        return
    if event_type == "media_input":
        logger.info("cartesia_media_input_ignored session_id=%s", session.session_id)
        return
    if event_type == "custom" and event.get("metadata", {}).get("transcript"):
        result = await process_transcript(session, event["metadata"]["transcript"])
        response_text = result.get("response_text") or result.get("response") or ""
        if response_text:
            await send_agent_response(websocket, session, response_text)
        return
    if event_type == "dtmf":
        await log_event(session.session_id, "dtmf", event)
        return
    raise HTTPException(status_code=400, detail=f"Unsupported event: {event_type}")
