from __future__ import annotations

import base64
from html import escape
import json
import logging
from typing import Any
from urllib.parse import urlencode
import uuid

from fastapi import APIRouter, Form, Request, Response, WebSocket

from app import config
from app.app_state import get_transcriber, sessions
from app.call_state import CallState
from app.call_state_manager import call_state_manager
from app.compliance import log_event
from app.prompts import PROMPT_VERSION
from app.audio import fail_safe_handoff, send_opening_greeting
from app.turn_handler import (
    cancel_session_tasks,
    finalize_call,
    schedule_response_task,
)

logger = logging.getLogger("voice_agent")

router = APIRouter()


@router.post("/twilio/voice")
async def twilio_voice_webhook(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(default=""),
    To: str = Form(default=""),
) -> Response:
    session_id = str(uuid.uuid4())
    await log_event(
        session_id,
        "twilio_call_started",
        {
            "call_sid": CallSid,
            "from": From,
            "to": To,
            "headers": dict(request.headers),
        },
    )
    stream_url = build_twilio_stream_url(request)
    twiml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
  <Connect>
    <Stream url="{escape(stream_url)}">
      <Parameter name="session_id" value="{escape(session_id)}" />
      <Parameter name="caller_id" value="{escape(From)}" />
      <Parameter name="called_number" value="{escape(To)}" />
    </Stream>
  </Connect>
</Response>"""
    return Response(content=twiml, media_type="application/xml")


@router.websocket("/ws/twilio-media")
async def twilio_media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session: CallState | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                logger.info(
                    "twilio_media_disconnect session_id=%s stream_sid=%s code=%s",
                    session.session_id if session is not None else None,
                    session.stream_sid if session is not None else None,
                    message.get("code"),
                )
                break
            raw = message.get("text")
            if raw is None:
                logger.info(
                    "twilio_media_nontext session_id=%s stream_sid=%s message_type=%s",
                    session.session_id if session is not None else None,
                    session.stream_sid if session is not None else None,
                    message["type"],
                )
                continue
            event = json.loads(raw)
            event_type = event.get("event")

            if event_type == "connected":
                logger.info("twilio_media_connected")
                continue

            if event_type == "start":
                start = event.get("start", {})
                params = start.get("customParameters", {})
                session_id = params.get("session_id") or str(uuid.uuid4())
                session = sessions.get(session_id) or call_state_manager.get(session_id) or call_state_manager.create(session_id, start.get("callSid"))
                session.prompt_version = PROMPT_VERSION
                sessions[session_id] = session
                session.stream_sid = event.get("streamSid")
                session.call_sid = start.get("callSid")
                session.twilio_websocket = websocket
                session.ink_stream = await get_transcriber().open_twilio_turn_stream(session, websocket)
                logger.info("twilio_media_start session_id=%s call_sid=%s stream_sid=%s", session.session_id, session.call_sid, session.stream_sid)
                await log_event(session.session_id, "twilio_stream_start", event)
                schedule_response_task(session, send_opening_greeting(websocket, session))
                continue

            if event_type == "media" and session is not None:
                payload = event["media"]["payload"]
                mulaw_chunk = base64.b64decode(payload)
                await get_transcriber().send_twilio_audio(session.ink_stream, mulaw_chunk)
                continue

            if event_type == "dtmf" and session is not None:
                logger.info("twilio_dtmf session_id=%s stream_sid=%s", session.session_id, session.stream_sid)
                await log_event(session.session_id, "twilio_dtmf", event)
                continue

            if event_type == "mark" and session is not None:
                mark_name = event.get("mark", {}).get("name")
                logger.info("MARK_COMPLETE [%s] mark=%s", session.session_id, mark_name)
                if session.last_mark == mark_name:
                    session.tts_playing = False
                    session.last_mark = None
                    session.response_buffer.active = False
                await log_event(session.session_id, "twilio_mark", event)
                continue

            if event_type == "stop" and session is not None:
                logger.info("twilio_media_stop session_id=%s stream_sid=%s", session.session_id, session.stream_sid)
                await log_event(session.session_id, "twilio_stream_stop", event)
                await cancel_session_tasks(session)
                await get_transcriber().close_turn_stream(session)
                await finalize_call(session, resolved=session.resolved)
                break
        if session is not None:
            await cancel_session_tasks(session)
            await get_transcriber().close_turn_stream(session)
            await log_event(session.session_id, "twilio_disconnect", {"session_id": session.session_id})
    except Exception as exc:
        logger.exception(
            "twilio_media_exception session_id=%s stream_sid=%s",
            session.session_id if session is not None else None,
            session.stream_sid if session is not None else None,
        )
        if session is not None:
            await cancel_session_tasks(session)
            await get_transcriber().close_turn_stream(session)
            await fail_safe_handoff(websocket, session, f"twilio_exception:{exc}", transport="twilio")
            await finalize_call(session, resolved=session.resolved)
        else:
            await websocket.close(code=1011)


def build_twilio_stream_url(request: Request) -> str:
    public_base = config.PUBLIC_BASE_URL
    if public_base:
        base = public_base.rstrip("/")
        if base.startswith("https://"):
            base = "wss://" + base.removeprefix("https://")
        elif base.startswith("http://"):
            base = "ws://" + base.removeprefix("http://")
        return f"{base}/ws/twilio-media"
    host = request.headers.get("host", "127.0.0.1:8000")
    scheme = "wss" if request.url.scheme == "https" else "ws"
    return f"{scheme}://{host}/ws/twilio-media"
