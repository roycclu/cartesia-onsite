from __future__ import annotations

import asyncio
import audioop
import base64
import json
import logging
import time
import uuid

from fastapi import WebSocket
import websockets

from app import config
from app.app_state import get_orchestrator, get_tts
from app.call_state import CallState
from app.compliance import log_event
from app.prompts import HUMAN_HANDOFF_PROMPT
from app.tools.insurance import trigger_handoff

logger = logging.getLogger("voice_agent")

TTS_URL = "wss://api.cartesia.ai/tts/websocket"
TWILIO_FRAME_SIZE = 160
TWILIO_FRAME_PACE_SECONDS = 0.018


class CartesiaTTS:
    def __init__(self) -> None:
        self.api_key = config.CARTESIA_API_KEY
        self.version = config.CARTESIA_VERSION
        self.voice_id = config.CARTESIA_VOICE_ID

    async def stream_synthesize(self, session_id: str, text: str):
        if not self.api_key:
            silence = b"\x00\x00" * 1600
            yield base64.b64encode(silence).decode("utf-8")
            return
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
        async with websockets.connect(TTS_URL, additional_headers=headers, max_size=8_000_000) as ws:
            await ws.send(json.dumps(payload))
            while True:
                raw = await ws.recv()
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                if message.get("type") == "chunk":
                    yield message["data"]
                if message.get("type") == "done":
                    break
                if message.get("type") == "error":
                    raise RuntimeError(message.get("message", "Cartesia TTS error"))

    async def synthesize(self, session_id: str, text: str) -> list[str]:
        return [chunk async for chunk in self.stream_synthesize(session_id, text)]


async def send_agent_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
    transport: str = "cartesia",
) -> None:
    if transport == "twilio":
        await send_twilio_response(websocket, session, text)
        return
    audio_chunks = await get_tts().synthesize(session.session_id, text)
    for chunk in audio_chunks:
        await websocket.send_json(
            {
                "event": "media_output",
                "stream_id": session.stream_sid,
                "media": {"payload": chunk},
                "text": text,
            }
        )


async def send_opening_greeting(websocket: WebSocket, session: CallState) -> None:
    greeting = await get_orchestrator().generate_greeting()
    await send_twilio_response(websocket, session, greeting)


async def log_response_started(session: CallState) -> None:
    if session.current_turn_id is None or session.current_turn_response_started_logged:
        return
    session.current_turn_response_started_logged = True
    latency_ms = None
    if session.current_turn_end_latency_t0 is not None:
        latency_ms = max(0, int((time.time() - session.current_turn_end_latency_t0) * 1000))
    logger.info(
        "LATENCY [%s] turn_id=%s latency_ms=%s",
        session.session_id,
        session.current_turn_id,
        latency_ms,
    )
    await log_event(
        session.session_id,
        "response_started",
        {"turn_id": session.current_turn_id, "latency_ms": latency_ms},
    )
    if session.current_turn_outcome is None:
        session.current_turn_outcome = "responded"
        await log_event(
            session.session_id,
            "turn_outcome",
            {"turn_id": session.current_turn_id, "outcome": "responded"},
        )


async def send_twilio_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
) -> None:
    async with session.twilio_send_lock:
        logger.info("TURN [%s] TTS: %s", session.session_id, text)
        session.tts_playing = True
        session.last_mark = None
        session.active_tts_task = asyncio.current_task()
        first_audio_logged = False
        mulaw_buffer = bytearray()
        try:
            try:
                async for chunk in get_tts().stream_synthesize(session.session_id, text):
                    pcm_chunk = base64.b64decode(chunk)
                    mulaw_buffer.extend(audioop.lin2ulaw(pcm_chunk, 2))
                    first_audio_logged = await send_audio_to_twilio(
                        websocket,
                        session,
                        bytes_buffer=mulaw_buffer,
                        first_audio_logged=first_audio_logged,
                    )
            except Exception as exc:
                if "402" in str(exc) or "quota" in str(exc).lower():
                    logger.error("TTS_QUOTA_EXCEEDED [%s]", session.session_id)
                    await handle_tts_unavailable(websocket, session)
                    return
                raise
            if mulaw_buffer:
                padded_frame = bytes(mulaw_buffer[:TWILIO_FRAME_SIZE]).ljust(TWILIO_FRAME_SIZE, b"\xff")
                if session.interrupted or session.response_buffer.superseded:
                    await send_clear_to_twilio(websocket, session)
                    return
                await send_twilio_media_frame(websocket, session.stream_sid, padded_frame)
                if not first_audio_logged:
                    await log_response_started(session)
                    first_audio_logged = True
                await asyncio.sleep(TWILIO_FRAME_PACE_SECONDS)
                mulaw_buffer.clear()
            mark_name = f"tts_{int(time.time() * 1000)}"
            session.last_mark = mark_name
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "mark",
                        "streamSid": session.stream_sid,
                        "mark": {"name": mark_name},
                    }
                )
            )
        finally:
            if session.active_tts_task is asyncio.current_task():
                session.active_tts_task = None


async def handle_tts_unavailable(websocket: WebSocket, session: CallState) -> None:
    await trigger_handoff(session.session_id, "tts_unavailable", "TTS unavailable")
    session.should_handoff = True
    session.handoff_reason = "tts_unavailable"
    session.tts_playing = False
    await send_clear_to_twilio(websocket, session)
    try:
        await websocket.close(code=1011)
    except RuntimeError:
        logger.info("tts_unavailable_socket_closed session_id=%s stream_sid=%s", session.session_id, session.stream_sid)


async def fail_safe_handoff(websocket: WebSocket, session: CallState, reason: str, transport: str = "cartesia") -> None:
    payload = await trigger_handoff(session.session_id, reason, "Automatic fallback handoff")
    session.should_handoff = True
    session.handoff_reason = reason
    if session.current_turn_id is not None and session.current_turn_outcome is None:
        session.current_turn_outcome = "handoff"
        await log_event(
            session.session_id,
            "turn_outcome",
            {"turn_id": session.current_turn_id, "outcome": "handoff", "reason": reason},
        )
    if transport == "twilio":
        try:
            await send_twilio_response(websocket, session, HUMAN_HANDOFF_PROMPT)
            await websocket.send_text(
                json.dumps(
                    {
                        "event": "mark",
                        "streamSid": session.stream_sid,
                        "mark": {"name": payload["reason_code"]},
                    }
                )
            )
            await websocket.close(code=1000)
        except RuntimeError:
            logger.info(
                "twilio_handoff_socket_closed session_id=%s stream_sid=%s reason=%s",
                session.session_id,
                session.stream_sid,
                payload["reason_code"],
            )
        return
    await websocket.send_json(
        {
            "event": "transfer_call",
            "stream_id": session.stream_sid,
            "reason": payload["reason_code"],
            "text": HUMAN_HANDOFF_PROMPT,
        }
    )
    await websocket.close(code=1011)


async def send_audio_to_twilio(
    websocket: WebSocket,
    session: CallState,
    *,
    bytes_buffer: bytearray,
    first_audio_logged: bool,
) -> bool:
    while len(bytes_buffer) >= TWILIO_FRAME_SIZE:
        if session.interrupted or session.response_buffer.superseded:
            await send_clear_to_twilio(websocket, session)
            return first_audio_logged
        frame = bytes(bytes_buffer[:TWILIO_FRAME_SIZE])
        del bytes_buffer[:TWILIO_FRAME_SIZE]
        await send_twilio_media_frame(websocket, session.stream_sid, frame)
        if not first_audio_logged:
            await log_response_started(session)
            first_audio_logged = True
        await asyncio.sleep(TWILIO_FRAME_PACE_SECONDS)
    return first_audio_logged


async def send_twilio_media_frame(websocket: WebSocket, stream_sid: str | None, frame: bytes) -> None:
    if stream_sid is None:
        return
    payload = base64.b64encode(frame).decode("utf-8")
    await websocket.send_text(
        json.dumps(
            {
                "event": "media",
                "streamSid": stream_sid,
                "media": {"payload": payload},
            }
        )
    )


async def send_clear_to_twilio(websocket: WebSocket, session: CallState) -> None:
    if session.stream_sid:
        await websocket.send_json({"event": "clear", "streamSid": session.stream_sid})
