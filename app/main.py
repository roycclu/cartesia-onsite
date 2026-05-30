from __future__ import annotations

import asyncio
import base64
import audioop
from datetime import datetime, timezone
from html import escape
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlencode
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import websockets

from app.compliance import get_session_log, log_event
from app.config import load_runtime_env
from app.db import close_db, database_status, init_db
from app.orchestration import GraphState, InsuranceOrchestrator
from app.tools import trigger_handoff


logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

STT_URL = "wss://api.cartesia.ai/stt/turns/websocket"
TTS_URL = "wss://api.cartesia.ai/tts/websocket"
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
APP_STARTED_AT = datetime.now(timezone.utc)
logger = logging.getLogger("voice_agent")

app = FastAPI(title="Insurance Voice Agent Demo", version="0.1.0")


@dataclass
class InkTurnStream:
    websocket: Any
    listener_task: asyncio.Task[None]


class TextTurnRequest(BaseModel):
    session_id: str | None = None
    transcript: str


@dataclass
class SessionContext:
    session_id: str
    stream_id: str | None = None
    call_sid: str | None = None
    verified: bool = False
    policy_number: str | None = None
    ssn_last4: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)
    human_requests: int = 0
    verification_attempts: int = 0
    ink_stream: InkTurnStream | None = None
    twilio_send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class CartesiaTranscriber:
    def __init__(self) -> None:
        self.api_key = os.getenv("CARTESIA_API_KEY")
        self.version = os.getenv("CARTESIA_VERSION", "2026-03-01")

    async def open_twilio_turn_stream(self, session: SessionContext, websocket: WebSocket) -> InkTurnStream | None:
        if not self.api_key:
            logger.warning("cartesia_ink_unavailable session_id=%s reason=missing_api_key", session.session_id)
            return None

        headers = {"X-API-Key": self.api_key}
        query = urlencode(
            {
                "model": "ink-2",
                "encoding": "pcm_mulaw",
                "sample_rate": "8000",
                "cartesia_version": self.version,
            }
        )
        stt_ws = await websockets.connect(f"{STT_URL}?{query}", additional_headers=headers, max_size=8_000_000)
        listener_task = asyncio.create_task(self._consume_turn_events(stt_ws, websocket, session))
        return InkTurnStream(websocket=stt_ws, listener_task=listener_task)

    async def send_twilio_audio(self, stream: InkTurnStream | None, chunk: bytes) -> None:
        if stream is None:
            return
        await stream.websocket.send(chunk)

    async def close_turn_stream(self, session: SessionContext) -> None:
        stream = session.ink_stream
        session.ink_stream = None
        if stream is None:
            return
        try:
            await stream.websocket.send(json.dumps({"type": "close"}))
        except Exception:
            pass
        try:
            await asyncio.wait_for(stream.listener_task, timeout=2)
        except Exception:
            stream.listener_task.cancel()
        try:
            await stream.websocket.close()
        except Exception:
            pass

    async def _consume_turn_events(self, stt_ws: Any, twilio_ws: WebSocket, session: SessionContext) -> None:
        try:
            async for raw in stt_ws:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                event_type = message.get("type")
                logger.info(
                    "cartesia_turn_event session_id=%s stream_sid=%s type=%s transcript=%s",
                    session.session_id,
                    session.stream_id,
                    event_type,
                    message.get("transcript"),
                )
                if event_type == "connected":
                    logger.info("cartesia_turn_stream_open session_id=%s stream_sid=%s", session.session_id, session.stream_id)
                    await log_event(session.session_id, "cartesia_stt_connected", message)
                    continue
                if event_type == "turn.start":
                    await log_event(session.session_id, "turn_start", message)
                    continue
                if event_type in {"turn.update", "turn.eager_end", "turn.resume"}:
                    await log_event(session.session_id, event_type.replace(".", "_"), message)
                    continue
                if event_type == "turn.end":
                    transcript = (message.get("transcript") or "").strip()
                    logger.info("TURN [%s] USER: %s", session.session_id, transcript)
                    await log_event(session.session_id, "asr_result", {"text": transcript, "provider": "cartesia_ink_2"})
                    if transcript:
                        result = await process_transcript(session, transcript)
                        await send_agent_response(twilio_ws, session, result["response_text"], transport="twilio")
                    continue
                if event_type == "error":
                    raise RuntimeError(message.get("message", "Cartesia Ink-2 error"))
        except Exception as exc:
            logger.exception(
                "cartesia_turn_stream_exception session_id=%s stream_sid=%s",
                session.session_id,
                session.stream_id,
            )
            await fail_safe_handoff(twilio_ws, session, f"cartesia_turns_exception:{exc}", transport="twilio")
        finally:
            logger.info("cartesia_turn_stream_close session_id=%s stream_sid=%s", session.session_id, session.stream_id)


class CartesiaTTS:
    def __init__(self) -> None:
        self.api_key = os.getenv("CARTESIA_API_KEY")
        self.version = os.getenv("CARTESIA_VERSION", "2026-03-01")
        self.voice_id = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")

    async def synthesize(self, session_id: str, text: str) -> list[str]:
        if not self.api_key:
            silence = b"\x00\x00" * 1600
            return [base64.b64encode(silence).decode("utf-8")]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": self.version,
        }
        payload = {
            "model_id": "sonic-latest",
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "language": "en",
            "context_id": str(uuid.uuid4()),
            "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 8000},
            "continue": False,
        }
        audio_chunks: list[str] = []
        async with websockets.connect(TTS_URL, additional_headers=headers, max_size=8_000_000) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message.get("type") == "chunk":
                    audio_chunks.append(message["data"])
                if message.get("type") == "done":
                    break
                if message.get("type") == "error":
                    raise RuntimeError(message.get("message", "Cartesia TTS error"))
        return audio_chunks


sessions: dict[str, SessionContext] = {}
orchestrator: InsuranceOrchestrator | None = None
transcriber: CartesiaTranscriber | None = None
tts: CartesiaTTS | None = None


@app.on_event("startup")
async def startup() -> None:
    global orchestrator, transcriber, tts
    load_runtime_env()
    await init_db()
    orchestrator = InsuranceOrchestrator()
    transcriber = CartesiaTranscriber()
    tts = CartesiaTTS()


@app.get("/health")
async def health() -> dict[str, str | float]:
    return {
        "status": "ok",
        "version": read_version(),
        "database": await database_status(),
        "uptime": round((datetime.now(timezone.utc) - APP_STARTED_AT).total_seconds(), 3),
    }


@app.on_event("shutdown")
async def shutdown() -> None:
    await close_db()


@app.post("/twilio/voice")
async def twilio_voice_webhook(
    request: Request,
    CallSid: str = Form(...),
    From: str = Form(default=""),
    To: str = Form(default=""),
) -> Response:
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionContext(session_id=session_id, call_sid=CallSid)
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


@app.post("/calls/start")
async def start_call(payload: dict[str, Any]) -> dict[str, str]:
    session_id = payload.get("session_id") or str(uuid.uuid4())
    sessions[session_id] = SessionContext(session_id=session_id)
    await log_event(session_id, "call_started", payload)
    return {"session_id": session_id}


@app.post("/demo/text-turn")
async def demo_text_turn(request: TextTurnRequest) -> dict[str, Any]:
    session_id = request.session_id or str(uuid.uuid4())
    session = sessions.setdefault(session_id, SessionContext(session_id=session_id))
    result = await process_transcript(session, request.transcript)
    return {
        "session_id": session_id,
        "response_text": result["response_text"],
        "verified": session.verified,
        "should_handoff": result.get("should_handoff", False),
        "handoff_reason": result.get("handoff_reason"),
    }


@app.get("/sessions/{session_id}/compliance-log")
async def session_log(session_id: str) -> list[dict[str, Any]]:
    return await get_session_log(session_id)


@app.websocket("/ws/cartesia/{session_id}")
async def cartesia_stream(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = sessions.setdefault(session_id, SessionContext(session_id=session_id))
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


@app.websocket("/ws/twilio-media")
async def twilio_media_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    session: SessionContext | None = None
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                logger.info(
                    "twilio_media_disconnect session_id=%s stream_sid=%s code=%s",
                    session.session_id if session is not None else None,
                    session.stream_id if session is not None else None,
                    message.get("code"),
                )
                break
            raw = message.get("text")
            if raw is None:
                logger.info(
                    "twilio_media_nontext session_id=%s stream_sid=%s message_type=%s",
                    session.session_id if session is not None else None,
                    session.stream_id if session is not None else None,
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
                session = sessions.setdefault(session_id, SessionContext(session_id=session_id))
                session.stream_id = event.get("streamSid")
                session.call_sid = start.get("callSid")
                session.ink_stream = await get_transcriber().open_twilio_turn_stream(session, websocket)
                logger.info("twilio_media_start session_id=%s call_sid=%s stream_sid=%s", session.session_id, session.call_sid, session.stream_id)
                await log_event(session.session_id, "twilio_stream_start", event)
                await send_twilio_response(websocket, session, "Thanks for calling. Please share your policy number and the last four digits of your Social Security number.")
                continue

            if event_type == "media" and session is not None:
                payload = event["media"]["payload"]
                mulaw_chunk = base64.b64decode(payload)
                await get_transcriber().send_twilio_audio(session.ink_stream, mulaw_chunk)
                continue

            if event_type == "dtmf" and session is not None:
                logger.info("twilio_dtmf session_id=%s stream_sid=%s", session.session_id, session.stream_id)
                await log_event(session.session_id, "twilio_dtmf", event)
                continue

            if event_type == "stop" and session is not None:
                logger.info("twilio_media_stop session_id=%s stream_sid=%s", session.session_id, session.stream_id)
                await log_event(session.session_id, "twilio_stream_stop", event)
                await get_transcriber().close_turn_stream(session)
                break
        if session is not None:
            await get_transcriber().close_turn_stream(session)
            await log_event(session.session_id, "twilio_disconnect", {"session_id": session.session_id})
    except Exception as exc:
        logger.exception(
            "twilio_media_exception session_id=%s stream_sid=%s",
            session.session_id if session is not None else None,
            session.stream_id if session is not None else None,
        )
        if session is not None:
            await get_transcriber().close_turn_stream(session)
            await fail_safe_handoff(websocket, session, f"twilio_exception:{exc}", transport="twilio")
        else:
            await websocket.close(code=1011)


async def handle_websocket_event(websocket: WebSocket, session: SessionContext, event: dict[str, Any]) -> None:
    event_type = event.get("event")
    if event_type == "start":
        session.stream_id = event.get("stream_id") or str(uuid.uuid4())
        await log_event(session.session_id, "stream_start", event)
        await websocket.send_json({"event": "ack", "stream_id": session.stream_id, "config": event.get("config", {})})
        return
    if event_type == "media_input":
        logger.info("cartesia_media_input_ignored session_id=%s", session.session_id)
        return
    if event_type == "custom" and event.get("metadata", {}).get("transcript"):
        result = await process_transcript(session, event["metadata"]["transcript"])
        await send_agent_response(websocket, session, result["response_text"])
        return
    if event_type == "dtmf":
        await log_event(session.session_id, "dtmf", event)
        return
    raise HTTPException(status_code=400, detail=f"Unsupported event: {event_type}")


async def process_transcript(session: SessionContext, transcript: str) -> GraphState:
    await log_event(session.session_id, "user_transcript", {"text": transcript})
    if any(word in transcript.lower() for word in ("human", "representative", "agent")):
        session.human_requests += 1
        if session.human_requests >= 2:
            payload = await trigger_handoff(session.session_id, "human_requested_twice", transcript)
            response_text = "I’m connecting you with a human representative now."
            await log_event(session.session_id, "llm_response", {"text": response_text, "handoff": True})
            return {
                "session_id": session.session_id,
                "transcript": transcript,
                "response_text": response_text,
                "should_handoff": True,
                "handoff_reason": "human_requested_twice",
                "tool_result": payload,
            }
    state: GraphState = {
        "session_id": session.session_id,
        "transcript": transcript,
        "history": session.history,
        "verified": session.verified,
        "policy_number": session.policy_number,
        "ssn_last4": session.ssn_last4,
        "verification_attempts": session.verification_attempts,
        "should_handoff": False,
        "handoff_reason": None,
        "llm_error": None,
    }
    result = await get_orchestrator().run_turn(state)
    session.verified = result.get("verified", session.verified)
    session.policy_number = result.get("policy_number", session.policy_number)
    session.ssn_last4 = result.get("ssn_last4", session.ssn_last4)
    session.verification_attempts = result.get("verification_attempts", session.verification_attempts)
    session.history.extend(
        [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": result["response_text"]},
        ]
    )
    return result


async def send_agent_response(websocket: WebSocket, session: SessionContext, text: str, transport: str = "cartesia") -> None:
    if transport == "twilio":
        await send_twilio_response(websocket, session, text)
        return
    audio_chunks = await get_tts().synthesize(session.session_id, text)
    for chunk in audio_chunks:
        await websocket.send_json(
            {
                "event": "media_output",
                "stream_id": session.stream_id,
                "media": {"payload": chunk},
                "text": text,
            }
        )


async def fail_safe_handoff(websocket: WebSocket, session: SessionContext, reason: str, transport: str = "cartesia") -> None:
    payload = await trigger_handoff(session.session_id, reason, "Automatic fallback handoff")
    if transport == "twilio":
        try:
            await send_twilio_response(websocket, session, "I’m transferring you to a human representative.")
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "mark",
                        "streamSid": session.stream_id,
                        "mark": {"name": payload["reason_code"]},
                    }
                )
            )
            await websocket.close(code=1000)
        except RuntimeError:
            logger.info(
                "twilio_handoff_socket_closed session_id=%s stream_sid=%s reason=%s",
                session.session_id,
                session.stream_id,
                payload["reason_code"],
            )
        return
    await websocket.send_json(
        {
            "event": "transfer_call",
            "stream_id": session.stream_id,
            "transfer": {"target_phone_number": os.getenv("HUMAN_HANDOFF_NUMBER", "+15555550199")},
            "reason": payload["reason_code"],
            "text": "I’m transferring you to a human representative.",
        }
    )
    await websocket.close(code=1011)


async def send_twilio_response(websocket: WebSocket, session: SessionContext, text: str) -> None:
    async with session.twilio_send_lock:
        logger.info("TURN [%s] TTS: %s", session.session_id, text)
        audio_chunks = await get_tts().synthesize(session.session_id, text)
        for chunk in audio_chunks:
            pcm_chunk = base64.b64decode(chunk)
            mulaw_chunk = audioop.lin2ulaw(pcm_chunk, 2)
            payload = base64.b64encode(mulaw_chunk).decode("utf-8")
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": session.stream_id,
                        "media": {"payload": payload},
                    }
                )
            )
        await websocket.send_text(
            json.dumps(
                {
                    "event": "mark",
                    "streamSid": session.stream_id,
                    "mark": {"name": f"response-{uuid.uuid4()}"},
                }
            )
        )


def build_twilio_stream_url(request: Request) -> str:
    public_base = os.getenv("PUBLIC_BASE_URL")
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


def read_version() -> str:
    try:
        return VERSION_FILE.read_text(encoding="utf-8").strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


def get_orchestrator() -> InsuranceOrchestrator:
    if orchestrator is None:
        raise RuntimeError("Application startup has not initialized the orchestrator.")
    return orchestrator


def get_transcriber() -> CartesiaTranscriber:
    if transcriber is None:
        raise RuntimeError("Application startup has not initialized the transcriber.")
    return transcriber


def get_tts() -> CartesiaTTS:
    if tts is None:
        raise RuntimeError("Application startup has not initialized TTS.")
    return tts


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
