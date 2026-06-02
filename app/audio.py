from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
import uuid
from contextlib import suppress

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


# Manages a single streaming TTS context for the active caller turn.
class CartesiaTTS:
    def __init__(self) -> None:
        self.api_key = config.CARTESIA_API_KEY
        self.version = config.CARTESIA_VERSION
        self.voice_id = config.CARTESIA_VOICE_ID

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Cartesia-Version": self.version,
        }

    async def start_turn_context(self, websocket: WebSocket, session: CallState) -> str | None:
        if session.current_tts_open and session.current_tts_context_id and session.current_tts_ws is not None:
            return session.current_tts_context_id
        if not self.api_key:
            return None
        headers = self._headers()
        tts_ws = await websockets.connect(TTS_URL, additional_headers=headers, max_size=8_000_000)
        context_id = str(uuid.uuid4())
        session.current_tts_context_id = context_id
        session.current_tts_ws = tts_ws
        session.current_tts_open = True
        session.current_tts_finalized = False
        session.current_tts_has_input = False
        session.tts_playing = True
        session.last_mark = None
        reader_task = asyncio.create_task(self._consume_turn_audio(websocket, session, tts_ws, context_id))
        session.current_tts_reader_task = reader_task
        session.active_tts_task = reader_task
        return context_id

    async def send_turn_text_chunk(
        self,
        websocket: WebSocket,
        session: CallState,
        text: str,
        *,
        continue_response: bool,
    ) -> None:
        context_id = await self.start_turn_context(websocket, session)
        if context_id is None:
            return
        if not text and continue_response:
            return
        payload = {
            "model_id": "sonic-latest",
            "transcript": text,
            "voice": {"mode": "id", "id": self.voice_id},
            "language": "en",
            "context_id": context_id,
            "output_format": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
            "continue": continue_response,
        }
        await session.current_tts_ws.send(json.dumps(payload))
        if text:
            session.current_tts_has_input = True
        if not continue_response:
            session.current_tts_finalized = True

    async def finalize_turn_context(self, session: CallState) -> None:
        if not session.current_tts_open or session.current_tts_ws is None:
            return
        if not session.current_tts_finalized:
            payload = {
                "model_id": "sonic-latest",
                "transcript": "",
                "voice": {"mode": "id", "id": self.voice_id},
                "language": "en",
                "context_id": session.current_tts_context_id,
                "output_format": {"container": "raw", "encoding": "pcm_mulaw", "sample_rate": 8000},
                "continue": False,
            }
            await session.current_tts_ws.send(json.dumps(payload))
            session.current_tts_finalized = True
        reader_task = session.current_tts_reader_task
        if reader_task is not None:
            await reader_task

    async def cancel_turn_context(self, session: CallState) -> None:
        tts_ws = session.current_tts_ws
        context_id = session.current_tts_context_id
        reader_task = session.current_tts_reader_task
        if tts_ws is None:
            session.current_tts_context_id = None
            session.current_tts_open = False
            session.current_tts_finalized = False
            session.current_tts_has_input = False
            return
        if context_id:
            with suppress(Exception):
                await tts_ws.send(json.dumps({"context_id": context_id, "cancel": True}))
        if reader_task is not None and not reader_task.done():
            reader_task.cancel()
            with suppress(asyncio.CancelledError):
                await reader_task
        with suppress(Exception):
            await tts_ws.close()
        if session.active_tts_task is reader_task:
            session.active_tts_task = None
        session.current_tts_ws = None
        session.current_tts_reader_task = None
        session.current_tts_context_id = None
        session.current_tts_open = False
        session.current_tts_finalized = False
        session.current_tts_has_input = False

    async def _consume_turn_audio(
        self,
        websocket: WebSocket,
        session: CallState,
        tts_ws: websockets.ClientConnection,
        context_id: str,
    ) -> None:
        first_audio_logged = False
        mulaw_buffer = bytearray()
        try:
            async with session.twilio_send_lock:
                while True:
                    raw = await tts_ws.recv()
                    if isinstance(raw, bytes):
                        continue
                    message = json.loads(raw)
                    message_type = message.get("type")
                    if message_type == "chunk":
                        mulaw_chunk = base64.b64decode(message["data"])
                        mulaw_buffer.extend(mulaw_chunk)
                        first_audio_logged = await send_audio_to_twilio(
                            websocket,
                            session,
                            bytes_buffer=mulaw_buffer,
                            first_audio_logged=first_audio_logged,
                        )
                        continue
                    if message_type == "done":
                        break
                    if message_type == "error":
                        raise RuntimeError(message.get("message", "Cartesia TTS error"))
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
                if first_audio_logged and session.stream_sid:
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
        except asyncio.CancelledError:
            if session.current_turn_response_started_logged or first_audio_logged:
                logger.info(
                    "TTS_CONTEXT_CANCELLED [%s] turn_id=%s context_id=%s",
                    session.session_id,
                    session.current_turn_id,
                    context_id,
                )
                return
            raise
        finally:
            with suppress(Exception):
                await tts_ws.close()
            if session.current_tts_context_id == context_id:
                session.current_tts_ws = None
                session.current_tts_reader_task = None
                session.current_tts_context_id = None
                session.current_tts_open = False
                session.current_tts_finalized = False
                session.current_tts_has_input = False
            if session.active_tts_task is asyncio.current_task():
                session.active_tts_task = None


async def send_agent_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
    *,
    continue_response: bool = True,
) -> None:
    await stream_twilio_response(websocket, session, text, continue_response=continue_response)


async def finalize_agent_response(
    websocket: WebSocket,
    session: CallState,
) -> None:
    await finalize_twilio_response(websocket, session)


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


    # Streams a partial assistant response into the active TTS turn context.
async def stream_twilio_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
    *,
    continue_response: bool,
) -> None:
    if not text:
        return
    logger.info("TURN [%s] TTS: %s", session.session_id, text)
    session.tts_playing = True
    session.last_mark = None
    try:
        await get_tts().send_turn_text_chunk(
            websocket,
            session,
            text,
            continue_response=continue_response,
        )
    except asyncio.CancelledError:
        if session.current_turn_response_started_logged:
            logger.info(
                "TTS_SEND_CANCELLED [%s] turn_id=%s response_started_logged=%s",
                session.session_id,
                session.current_turn_id,
                session.current_turn_response_started_logged,
            )
            return
        raise
    except Exception as exc:
        if "402" in str(exc) or "quota" in str(exc).lower():
            logger.error("TTS_QUOTA_EXCEEDED [%s]", session.session_id)
            await handle_tts_unavailable(websocket, session)
            return
        raise


async def finalize_twilio_response(websocket: WebSocket, session: CallState) -> None:
    if not session.current_tts_open:
        return
    await get_tts().finalize_turn_context(session)


async def send_twilio_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
) -> None:
    await stream_twilio_response(websocket, session, text, continue_response=False)
    await finalize_twilio_response(websocket, session)


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


    # Falls back to a human handoff when audio or turn processing cannot recover safely.
async def fail_safe_handoff(websocket: WebSocket, session: CallState, reason: str) -> None:
    payload = await trigger_handoff(session.session_id, reason, "Automatic fallback handoff")
    session.should_handoff = True
    session.handoff_reason = reason
    await get_tts().cancel_turn_context(session)
    if session.current_turn_id is not None and session.current_turn_outcome is None:
        session.current_turn_outcome = "handoff"
        await log_event(
            session.session_id,
            "turn_outcome",
            {"turn_id": session.current_turn_id, "outcome": "handoff", "reason": reason},
        )
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
        try:
            await asyncio.sleep(TWILIO_FRAME_PACE_SECONDS)
        except asyncio.CancelledError:
            if session.current_turn_response_started_logged or first_audio_logged:
                logger.info(
                    "TTS_FRAME_PACING_CANCELLED [%s] turn_id=%s response_started_logged=%s",
                    session.session_id,
                    session.current_turn_id,
                    session.current_turn_response_started_logged,
                )
                return first_audio_logged
            raise
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
