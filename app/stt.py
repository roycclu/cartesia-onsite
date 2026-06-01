from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import urlencode

from fastapi import WebSocket
import websockets

from app import config
from app.call_state import CallState
from app.compliance import log_event
from app.turn_handler import (
    handle_completed_turn,
    interrupt_twilio_playback,
    maybe_prefetch_from_partial,
    resolve_speculative_turn,
    schedule_response_task,
    should_process_transcript,
    start_speculative_task,
)

logger = logging.getLogger("voice_agent")

STT_URL = "wss://api.cartesia.ai/stt/turns/websocket"


class InkTurnStream:
    def __init__(self, websocket: Any, listener_task: asyncio.Task[None]) -> None:
        self.websocket = websocket
        self.listener_task = listener_task


class CartesiaTranscriber:
    def __init__(self) -> None:
        self.api_key = config.CARTESIA_API_KEY
        self.version = config.CARTESIA_VERSION

    async def open_twilio_turn_stream(self, session: CallState, websocket: WebSocket) -> InkTurnStream | None:
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

    async def close_turn_stream(self, session: CallState) -> None:
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

    async def _consume_turn_events(self, stt_ws: Any, twilio_ws: WebSocket, session: CallState) -> None:
        from app.audio import fail_safe_handoff

        try:
            async for raw in stt_ws:
                if isinstance(raw, bytes):
                    continue
                message = json.loads(raw)
                event_type = message.get("type")
                logger.info(
                    "cartesia_turn_event session_id=%s stream_sid=%s type=%s transcript=%s",
                    session.session_id,
                    session.stream_sid,
                    event_type,
                    message.get("transcript"),
                )
                if event_type == "connected":
                    logger.info("cartesia_turn_stream_open session_id=%s stream_sid=%s", session.session_id, session.stream_sid)
                    await log_event(session.session_id, "cartesia_stt_connected", message)
                    continue
                if event_type == "turn.start":
                    turn_id = session.start_new_turn()
                    await interrupt_twilio_playback(twilio_ws, session)
                    await log_event(session.session_id, "turn_start", {**message, "turn_id": turn_id})
                    continue
                if event_type in {"turn.update", "turn.resume"}:
                    transcript = (message.get("transcript") or "").strip()
                    if transcript:
                        await maybe_prefetch_from_partial(session, transcript)
                    await log_event(
                        session.session_id,
                        event_type.replace(".", "_"),
                        {**message, "turn_id": session.current_turn_id},
                    )
                    continue
                if event_type == "turn.eager_end":
                    await log_event(session.session_id, "turn_eager_end", {**message, "turn_id": session.current_turn_id})
                    transcript = (message.get("transcript") or "").strip()
                    if transcript and should_process_transcript(transcript):
                        t0 = time.time()
                        session.current_eager_latency_t0 = t0
                        session.pending_transcript = transcript
                        start_speculative_task(session, twilio_ws, transcript, t0)
                    continue
                if event_type == "turn.end":
                    transcript = (message.get("transcript") or "").strip()
                    session.current_turn_end_latency_t0 = time.time()
                    logger.info("TURN [%s] USER: %s", session.session_id, transcript)
                    await log_event(
                        session.session_id,
                        "asr_result",
                        {"text": transcript, "provider": "cartesia_ink_2", "turn_id": session.current_turn_id},
                    )
                    if transcript and should_process_transcript(transcript):
                        if await resolve_speculative_turn(session, twilio_ws, transcript):
                            continue
                        session.pending_transcript = transcript
                        schedule_response_task(
                            session,
                            handle_completed_turn(session, twilio_ws, transcript, time.time(), speculative=False),
                        )
                    elif transcript:
                        logger.info("SKIP_NOISE_TURN [%s] transcript=%r", session.session_id, transcript)
                    continue
                if event_type == "error":
                    raise RuntimeError(message.get("message", "Cartesia Ink-2 error"))
        except Exception as exc:
            logger.exception(
                "cartesia_turn_stream_exception session_id=%s stream_sid=%s",
                session.session_id,
                session.stream_sid,
            )
            await fail_safe_handoff(twilio_ws, session, f"cartesia_turns_exception:{exc}", transport="twilio")
        finally:
            logger.info("cartesia_turn_stream_close session_id=%s stream_sid=%s", session.session_id, session.stream_sid)
