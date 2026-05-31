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
import time
from urllib.parse import urlencode
import uuid
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import websockets

from app.call_state import CallState
from app.call_state_manager import call_state_manager
from app.compliance import get_session_log, log_event, persist_call_record
from app import config
from app.orchestration import GraphState, InsuranceOrchestrator, predict_intent_fast
from app.prompts import (
    GREETING_PROMPT,
    HUMAN_HANDOFF_PROMPT,
    HUMAN_REQUESTED_TWICE_PROMPT,
    PROMPT_VERSION,
)
from app.tools import get_claim_status, get_policy_info, trigger_handoff
from mock_data.db import close_db, database_status, ensure_pool, init_db


logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("websockets.client").setLevel(logging.WARNING)
logging.getLogger("websockets.server").setLevel(logging.WARNING)

STT_URL = "wss://api.cartesia.ai/stt/turns/websocket"
TTS_URL = "wss://api.cartesia.ai/tts/websocket"
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
APP_STARTED_AT = datetime.now(timezone.utc)
TWILIO_FRAME_SIZE = 160
TWILIO_FRAME_PACE_SECONDS = 0.018
logger = logging.getLogger("voice_agent")

app = FastAPI(title="Insurance Voice Agent Demo", version="0.1.0")


class InkTurnStream:
    def __init__(self, websocket: Any, listener_task: asyncio.Task[None]) -> None:
        self.websocket = websocket
        self.listener_task = listener_task


class TextTurnRequest(BaseModel):
    session_id: str | None = None
    transcript: str


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
                    await interrupt_twilio_playback(twilio_ws, session)
                    await log_event(session.session_id, "turn_start", message)
                    continue
                if event_type in {"turn.update", "turn.resume"}:
                    transcript = (message.get("transcript") or "").strip()
                    if transcript:
                        await maybe_prefetch_from_partial(session, transcript)
                    await log_event(session.session_id, event_type.replace(".", "_"), message)
                    continue
                if event_type == "turn.eager_end":
                    await log_event(session.session_id, "turn_eager_end", message)
                    transcript = (message.get("transcript") or "").strip()
                    if transcript:
                        t0 = time.time()
                        session.pending_transcript = transcript
                        start_speculative_task(session, twilio_ws, transcript, t0)
                    continue
                if event_type == "turn.end":
                    transcript = (message.get("transcript") or "").strip()
                    logger.info("TURN [%s] USER: %s", session.session_id, transcript)
                    await log_event(session.session_id, "asr_result", {"text": transcript, "provider": "cartesia_ink_2"})
                    if transcript:
                        if await resolve_speculative_turn(session, twilio_ws, transcript):
                            continue
                        session.pending_transcript = transcript
                        schedule_response_task(
                            session,
                            handle_completed_turn(session, twilio_ws, transcript, time.time(), speculative=False),
                        )
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


sessions: dict[str, CallState] = {}
orchestrator: InsuranceOrchestrator | None = None
transcriber: CartesiaTranscriber | None = None
tts: CartesiaTTS | None = None


@app.on_event("startup")
async def startup() -> None:
    global orchestrator, transcriber, tts
    config.load_runtime_env()
    logger.info("CONFIG_OK model=%s region=%s prompt_version=%s", config.OPENAI_MODEL, config.AWS_REGION, PROMPT_VERSION)
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
    state = call_state_manager.create(session_id, payload.get("call_sid"))
    state.prompt_version = PROMPT_VERSION
    sessions[session_id] = state
    await log_event(session_id, "call_started", payload)
    return {"session_id": session_id}


@app.post("/demo/text-turn")
async def demo_text_turn(request: TextTurnRequest) -> dict[str, Any]:
    session_id = request.session_id or str(uuid.uuid4())
    session = sessions.get(session_id) or call_state_manager.get(session_id) or call_state_manager.create(session_id, None)
    session.prompt_version = PROMPT_VERSION
    sessions[session_id] = session
    result = await process_transcript(session, request.transcript)
    return {
        "session_id": session_id,
        "response_text": result["response_text"],
        "verified": session.verified,
        "should_handoff": session.should_handoff,
        "handoff_reason": session.handoff_reason,
    }


@app.get("/sessions/{session_id}/compliance-log")
async def session_log(session_id: str) -> list[dict[str, Any]]:
    return await get_session_log(session_id)


@app.websocket("/ws/cartesia/{session_id}")
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


@app.websocket("/ws/twilio-media")
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
                schedule_response_task(
                    session,
                    send_twilio_response(
                        websocket,
                        session,
                        GREETING_PROMPT,
                    ),
                )
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


async def handle_websocket_event(websocket: WebSocket, session: CallState, event: dict[str, Any]) -> None:
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
        await send_agent_response(websocket, session, result["response_text"])
        return
    if event_type == "dtmf":
        await log_event(session.session_id, "dtmf", event)
        return
    raise HTTPException(status_code=400, detail=f"Unsupported event: {event_type}")


async def process_transcript(session: CallState, transcript: str) -> GraphState:
    await log_event(session.session_id, "user_transcript", {"text": transcript})
    if any(word in transcript.lower() for word in ("human", "representative", "agent")):
        session.human_requests += 1
        if session.human_requests >= 2:
            payload = await trigger_handoff(session.session_id, "human_requested_twice", transcript)
            session.should_handoff = True
            session.handoff_reason = "human_requested_twice"
            response_text = HUMAN_REQUESTED_TWICE_PROMPT
            await log_event(session.session_id, "llm_response", {"text": response_text, "handoff": True})
            session.add_turn("user", transcript)
            session.add_turn("assistant", response_text)
            return {
                "session_id": session.session_id,
                "transcript": transcript,
                "call_state": session,
                "response_text": response_text,
                "should_handoff": True,
                "handoff_reason": "human_requested_twice",
                "tool_result": payload,
            }
    state: GraphState = {
        "session_id": session.session_id,
        "transcript": transcript,
        "call_state": session,
        "should_handoff": session.should_handoff,
        "handoff_reason": session.handoff_reason,
        "llm_error": None,
    }
    if session.twilio_websocket is not None:
        state["sentence_handler"] = build_sentence_handler(session)
    result = await get_orchestrator().run_turn(state)
    session.add_turn("user", transcript)
    session.add_turn("assistant", result["response_text"])
    return result


async def send_agent_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
    transport: str = "cartesia",
    *,
    latency_t0: float | None = None,
) -> None:
    if transport == "twilio":
        await send_twilio_response(websocket, session, text, latency_t0=latency_t0)
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


async def fail_safe_handoff(websocket: WebSocket, session: CallState, reason: str, transport: str = "cartesia") -> None:
    payload = await trigger_handoff(session.session_id, reason, "Automatic fallback handoff")
    session.should_handoff = True
    session.handoff_reason = reason
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


async def send_twilio_response(
    websocket: WebSocket,
    session: CallState,
    text: str,
    *,
    latency_t0: float | None = None,
) -> None:
    async with session.twilio_send_lock:
        logger.info("TURN [%s] TTS: %s", session.session_id, text)
        session.tts_playing = True
        session.last_mark = None
        session.active_tts_task = asyncio.current_task()
        first_audio_logged = False
        mulaw_buffer = bytearray()
        try:
            async for chunk in get_tts().stream_synthesize(session.session_id, text):
                pcm_chunk = base64.b64decode(chunk)
                mulaw_buffer.extend(audioop.lin2ulaw(pcm_chunk, 2))
                first_audio_logged = await send_audio_to_twilio(
                    websocket,
                    session,
                    bytes_buffer=mulaw_buffer,
                    first_audio_logged=first_audio_logged,
                    latency_t0=latency_t0,
                )
            if mulaw_buffer:
                padded_frame = bytes(mulaw_buffer[:TWILIO_FRAME_SIZE]).ljust(TWILIO_FRAME_SIZE, b"\xff")
                if session.interrupted or session.response_buffer.superseded:
                    await send_clear_to_twilio(websocket, session)
                    return
                await send_twilio_media_frame(websocket, session.stream_sid, padded_frame)
                if latency_t0 is not None and not first_audio_logged:
                    logger.info("LATENCY [%s] eager_end_to_first_audio_ms=%s", session.session_id, int((time.time() - latency_t0) * 1000))
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


async def handle_completed_turn(
    session: CallState,
    websocket: WebSocket,
    transcript: str,
    latency_t0: float,
    *,
    speculative: bool,
) -> None:
    session.interrupted = False
    session.response_buffer.start()
    session.current_latency_t0 = latency_t0
    result = await process_transcript(session, transcript)
    logger.info("LATENCY [%s] llm_response_ms=%s", session.session_id, int((time.time() - latency_t0) * 1000))
    if session.twilio_websocket is not None and session.response_buffer.sentences_sent == 0:
        await send_agent_response(websocket, session, result["response_text"], transport="twilio", latency_t0=latency_t0)
    session.pending_transcript = None
    if speculative:
        session.speculative_transcript = transcript
        session.speculative_task = asyncio.current_task()
    if session.should_close:
        await finalize_call(session, resolved=session.resolved)
        await websocket.close(code=1000)


def schedule_response_task(session: CallState, coroutine: Any) -> None:
    if session.active_response_task is not None and not session.active_response_task.done():
        session.active_response_task.cancel()
    task = asyncio.create_task(coroutine)
    session.active_response_task = task

    def _clear_task(done_task: asyncio.Task[Any]) -> None:
        if session.active_response_task is done_task:
            session.active_response_task = None

    task.add_done_callback(_clear_task)


async def cancel_session_tasks(session: CallState) -> None:
    tasks = [session.active_response_task, session.active_tts_task, session.speculative_task]
    session.active_response_task = None
    session.active_tts_task = None
    session.speculative_task = None
    session.pending_transcript = None
    session.tts_playing = False
    session.last_mark = None
    session.interrupted = True
    session.response_buffer.supersede()
    session.current_latency_t0 = None
    for task in tasks:
        if task is not None and not task.done():
            task.cancel()
    for task in tasks:
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("session_task_cancel_exception session_id=%s", session.session_id)


async def interrupt_twilio_playback(websocket: WebSocket, session: CallState) -> None:
    if not session.tts_playing and session.active_response_task is None and session.speculative_task is None:
        return
    session.tts_playing = False
    session.last_mark = None
    session.interrupted = True
    session.response_buffer.supersede()
    if session.stream_sid:
        await send_clear_to_twilio(websocket, session)
        logger.info("BARGE_IN [%s] cleared Twilio buffer", session.session_id)
    await cancel_session_tasks(session)


async def send_audio_to_twilio(
    websocket: WebSocket,
    session: CallState,
    *,
    bytes_buffer: bytearray,
    first_audio_logged: bool,
    latency_t0: float | None,
) -> bool:
    while len(bytes_buffer) >= TWILIO_FRAME_SIZE:
        if session.interrupted or session.response_buffer.superseded:
            await send_clear_to_twilio(websocket, session)
            return first_audio_logged
        frame = bytes(bytes_buffer[:TWILIO_FRAME_SIZE])
        del bytes_buffer[:TWILIO_FRAME_SIZE]
        await send_twilio_media_frame(websocket, session.stream_sid, frame)
        if latency_t0 is not None and not first_audio_logged:
            logger.info("LATENCY [%s] eager_end_to_first_audio_ms=%s", session.session_id, int((time.time() - latency_t0) * 1000))
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


def start_speculative_task(session: CallState, websocket: WebSocket, transcript: str, latency_t0: float) -> None:
    if session.speculative_task is not None and not session.speculative_task.done():
        session.speculative_task.cancel()
    session.speculative_transcript = transcript
    task = asyncio.create_task(handle_completed_turn(session, websocket, transcript, latency_t0, speculative=True))
    session.speculative_task = task

    def _clear_speculative(done_task: asyncio.Task[Any]) -> None:
        if session.speculative_task is done_task:
            session.speculative_task = None

    task.add_done_callback(_clear_speculative)


async def resolve_speculative_turn(session: CallState, websocket: WebSocket, transcript: str) -> bool:
    speculative_transcript = session.speculative_transcript
    speculative_task = session.speculative_task
    if not speculative_transcript:
        return False
    similarity = transcript_similarity(speculative_transcript, transcript)
    hit = similarity >= 0.80
    logger.info("SPECULATIVE [%s] similarity=%.2f outcome=%s", session.session_id, similarity, "hit" if hit else "miss")
    if hit:
        session.pending_transcript = None
        session.speculative_transcript = None
        return True
    session.response_buffer.supersede()
    session.interrupted = True
    if speculative_task is not None and not speculative_task.done():
        speculative_task.cancel()
        try:
            await speculative_task
        except asyncio.CancelledError:
            pass
    await send_clear_to_twilio(websocket, session)
    session.speculative_task = None
    session.speculative_transcript = None
    session.pending_transcript = None
    session.interrupted = False
    return False


def transcript_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def build_sentence_handler(session: CallState):
    async def _handle_sentence(sentence: str) -> None:
        if session.response_buffer.superseded or session.interrupted:
            return
        await send_agent_response(
            session.twilio_websocket,
            session,
            sentence,
            transport="twilio",
            latency_t0=session.current_latency_t0,
        )
        session.current_latency_t0 = None
        session.response_buffer.sentence_complete()

    return _handle_sentence


async def maybe_prefetch_from_partial(session: CallState, transcript: str) -> None:
    if not session.verified:
        return
    if len(transcript.split()) < 4:
        return
    intent = predict_intent_fast(transcript)
    if intent is None or intent in session.prefetched_data or intent in session.prefetch_tasks:
        return

    async def _prefetch_result() -> None:
        try:
            if intent == "get_claim_status":
                result = await get_claim_status(session.session_id, session.policy_number or "")
            else:
                result = await get_policy_info(session.session_id, session.policy_number or "")
            session.prefetched_data[intent] = result
        finally:
            session.prefetch_tasks.pop(intent, None)

    session.prefetch_tasks[intent] = asyncio.create_task(_prefetch_result())


async def finalize_call(session: CallState, resolved: bool = False) -> None:
    final_state = call_state_manager.end(session.session_id or session.call_id, resolved=resolved)
    if final_state is None:
        return
    sessions.pop(final_state.session_id or "", None)
    await persist_call_record(final_state, await ensure_pool())


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
