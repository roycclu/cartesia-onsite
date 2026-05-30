from __future__ import annotations

import asyncio
import base64
import audioop
from html import escape
import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import websockets

from compliance import get_session_log, log_event
from db import init_db
from orchestration import GraphState, InsuranceOrchestrator
from tools import trigger_handoff
from vad import TurnDetector


CARTESIA_VERSION = os.getenv("CARTESIA_VERSION", "2026-03-01")
STT_URL = "wss://api.cartesia.ai/stt/websocket"
TTS_URL = "wss://api.cartesia.ai/tts/websocket"
DEFAULT_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091")

app = FastAPI(title="Insurance Voice Agent Demo", version="0.1.0")


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
    audio_buffer: bytearray = field(default_factory=bytearray)
    silence_timeouts: int = 0
    asr_failures: int = 0
    human_requests: int = 0
    heard_speech_in_turn: bool = False
    turn_detector: TurnDetector = field(default_factory=lambda: TurnDetector(sample_rate=8000))


class CartesiaTranscriber:
    def __init__(self) -> None:
        self.api_key = os.getenv("CARTESIA_API_KEY")

    async def transcribe(self, audio_bytes: bytes) -> dict[str, Any]:
        if not self.api_key:
            transcript = "<mock audio transcript unavailable>"
            confidence = 0.25 if not audio_bytes else 0.9
            return {"text": transcript, "confidence": confidence, "provider": "mock"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": CARTESIA_VERSION,
        }
        async with websockets.connect(STT_URL, additional_headers=headers, max_size=8_000_000) as ws:
            await ws.send(audio_bytes)
            await ws.send("finalize")
            text_parts: list[str] = []
            final_received = False
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message.get("type") == "transcript":
                    text_parts.append(message.get("text", ""))
                    final_received = final_received or message.get("is_final", False)
                if message.get("type") == "done":
                    break
                if message.get("type") == "error":
                    raise RuntimeError(message.get("message", "Cartesia STT error"))
            transcript = " ".join(part.strip() for part in text_parts if part.strip())
            confidence = 0.9 if final_received and transcript else 0.35
            await ws.send("close")
            return {"text": transcript, "confidence": confidence, "provider": "cartesia"}


class CartesiaTTS:
    def __init__(self) -> None:
        self.api_key = os.getenv("CARTESIA_API_KEY")
        self.voice_id = DEFAULT_VOICE_ID

    async def synthesize(self, session_id: str, text: str) -> list[str]:
        if not self.api_key:
            silence = b"\x00\x00" * 1600
            return [base64.b64encode(silence).decode("utf-8")]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": CARTESIA_VERSION,
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
orchestrator = InsuranceOrchestrator()
transcriber = CartesiaTranscriber()
tts = CartesiaTTS()


@app.on_event("startup")
async def startup() -> None:
    await init_db()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


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
                await handle_audio_chunk(websocket, session, message["bytes"])
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
            raw = await websocket.receive_text()
            event = json.loads(raw)
            event_type = event.get("event")

            if event_type == "connected":
                continue

            if event_type == "start":
                start = event.get("start", {})
                params = start.get("customParameters", {})
                session_id = params.get("session_id") or str(uuid.uuid4())
                session = sessions.setdefault(session_id, SessionContext(session_id=session_id))
                session.stream_id = event.get("streamSid")
                session.call_sid = start.get("callSid")
                session.turn_detector.sample_rate = 8000
                session.turn_detector.reset()
                await log_event(session.session_id, "twilio_stream_start", event)
                await send_twilio_response(websocket, session, "Thanks for calling. Please share your policy number and the last four digits of your Social Security number.")
                continue

            if event_type == "media" and session is not None:
                payload = event["media"]["payload"]
                mulaw_chunk = base64.b64decode(payload)
                pcm_chunk = audioop.ulaw2lin(mulaw_chunk, 2)
                await handle_audio_chunk(websocket, session, pcm_chunk, transport="twilio")
                continue

            if event_type == "dtmf" and session is not None:
                await log_event(session.session_id, "twilio_dtmf", event)
                continue

            if event_type == "stop" and session is not None:
                await log_event(session.session_id, "twilio_stream_stop", event)
                break
    except WebSocketDisconnect:
        if session is not None:
            await log_event(session.session_id, "twilio_disconnect", {"session_id": session.session_id})
    except Exception as exc:
        if session is not None:
            await fail_safe_handoff(websocket, session, f"twilio_exception:{exc}", transport="twilio")
        else:
            await websocket.close(code=1011)


async def handle_websocket_event(websocket: WebSocket, session: SessionContext, event: dict[str, Any]) -> None:
    event_type = event.get("event")
    if event_type == "start":
        session.stream_id = event.get("stream_id") or str(uuid.uuid4())
        session.turn_detector.reset()
        await log_event(session.session_id, "stream_start", event)
        await websocket.send_json({"event": "ack", "stream_id": session.stream_id, "config": event.get("config", {})})
        return
    if event_type == "media_input":
        payload = base64.b64decode(event["media"]["payload"])
        await handle_audio_chunk(websocket, session, payload)
        return
    if event_type == "custom" and event.get("metadata", {}).get("transcript"):
        result = await process_transcript(session, event["metadata"]["transcript"])
        await send_agent_response(websocket, session, result["response_text"])
        return
    if event_type == "dtmf":
        await log_event(session.session_id, "dtmf", event)
        return
    raise HTTPException(status_code=400, detail=f"Unsupported event: {event_type}")


async def handle_audio_chunk(websocket: WebSocket, session: SessionContext, chunk: bytes, transport: str = "cartesia") -> None:
    vad_result = session.turn_detector.ingest(chunk)
    if vad_result.speech_detected:
        session.heard_speech_in_turn = True
        session.silence_timeouts = 0
    if session.heard_speech_in_turn:
        session.audio_buffer.extend(chunk)
    if vad_result.end_of_turn:
        if not session.heard_speech_in_turn:
            session.turn_detector.reset()
            await handle_silence_timeout(websocket, session, transport=transport)
            return
        audio = bytes(session.audio_buffer)
        session.audio_buffer.clear()
        session.heard_speech_in_turn = False
        session.turn_detector.reset()
        asr_result = await transcriber.transcribe(audio)
        await log_event(session.session_id, "asr_result", asr_result)
        if asr_result["confidence"] < 0.6:
            session.asr_failures += 1
            if session.asr_failures >= 2:
                await fail_safe_handoff(websocket, session, "asr_low_confidence", transport=transport)
            else:
                await send_agent_response(websocket, session, "I didn’t catch that clearly. Please repeat your request.", transport=transport)
            return
        session.asr_failures = 0
        result = await process_transcript(session, asr_result["text"])
        await send_agent_response(websocket, session, result["response_text"], transport=transport)


async def handle_silence_timeout(websocket: WebSocket, session: SessionContext, transport: str = "cartesia") -> None:
    session.silence_timeouts += 1
    session.heard_speech_in_turn = False
    session.audio_buffer.clear()
    if session.silence_timeouts >= 2:
        await fail_safe_handoff(websocket, session, "silence_timeout", transport=transport)
        return
    await send_agent_response(websocket, session, "I’m still here. Please let me know how I can help with your insurance policy or claim.", transport=transport)


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
        "should_handoff": False,
        "handoff_reason": None,
        "llm_error": None,
    }
    result = await orchestrator.run_turn(state)
    session.verified = result.get("verified", session.verified)
    session.policy_number = result.get("policy_number", session.policy_number)
    session.ssn_last4 = result.get("ssn_last4", session.ssn_last4)
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
    audio_chunks = await tts.synthesize(session.session_id, text)
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
    audio_chunks = await tts.synthesize(session.session_id, text)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
