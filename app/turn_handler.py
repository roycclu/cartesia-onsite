from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import WebSocket

from app.app_state import get_orchestrator, sessions
from app.call_state import CallState
from app.call_state_manager import call_state_manager
from app.compliance import log_event, persist_call_record
from app.orchestration import GraphState
from app.prompts import HUMAN_REQUESTED_TWICE_PROMPT
from app.tools.insurance import get_claim_status, get_policy_info, trigger_handoff
from evals.llm_evals import run_post_call_evals
from mock_data.db import ensure_pool

logger = logging.getLogger("voice_agent")


def should_process_transcript(transcript: str) -> bool:
    cleaned = transcript.strip().strip(".,!?")
    return len(cleaned) >= 3


async def log_turn_outcome(session: CallState, outcome: str) -> None:
    if session.current_turn_id is None or session.current_turn_outcome is not None:
        return
    session.current_turn_outcome = outcome
    await log_event(
        session.session_id,
        "turn_outcome",
        {"turn_id": session.current_turn_id, "outcome": outcome},
    )


async def process_transcript(session: CallState, transcript: str) -> GraphState:
    await log_event(session.session_id, "user_transcript", {"text": transcript})
    if any(word in transcript.lower() for word in ("human", "representative", "agent")):
        session.human_requests += 1
        if session.human_requests >= 2:
            payload = await trigger_handoff(session.session_id, "human_requested_twice", transcript)
            session.should_handoff = True
            session.handoff_reason = "human_requested_twice"
            await log_turn_outcome(session, "handoff")
            response_text = HUMAN_REQUESTED_TWICE_PROMPT
            await log_event(
                session.session_id,
                "llm_response",
                {"text": response_text, "handoff": True, "turn_id": session.current_turn_id},
            )
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
    response_text = result.get("response_text") or result.get("response") or ""
    if response_text:
        session.add_turn("assistant", response_text)
    return result


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
    from app.app_state import get_tts

    tasks = [session.active_response_task, session.active_tts_task, session.speculative_task]
    session.active_response_task = None
    session.active_tts_task = None
    session.speculative_task = None
    session.pending_transcript = None
    session.tts_playing = False
    session.last_mark = None
    session.interrupted = True
    session.response_buffer.supersede()
    await get_tts().cancel_turn_context(session)
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


async def finalize_call(session: CallState, resolved: bool = False) -> None:
    final_state = call_state_manager.end(session.session_id or session.call_id, resolved=resolved)
    if final_state is None:
        return
    sessions.pop(final_state.session_id or "", None)
    await persist_call_record(final_state, await ensure_pool())
    asyncio.create_task(run_post_call_evals(final_state))


def start_speculative_task(session: CallState, websocket: WebSocket, transcript: str, latency_t0: float) -> None:
    from app.audio import fail_safe_handoff, send_agent_response  # noqa: F401

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
    from app.audio import send_clear_to_twilio

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
    await log_turn_outcome(session, "superseded")
    return False


def transcript_similarity(a: str, b: str) -> float:
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / max(len(words_a), len(words_b))


def build_sentence_handler(session: CallState):
    async def _handle_sentence(sentence: str) -> None:
        from app.audio import send_agent_response

        if session.response_buffer.superseded or session.interrupted:
            return
        await send_agent_response(
            session.twilio_websocket,
            session,
            sentence,
            transport="twilio",
            continue_response=True,
        )
        session.response_buffer.sentence_complete()

    return _handle_sentence


async def handle_completed_turn(
    session: CallState,
    websocket: WebSocket,
    transcript: str,
    latency_t0: float,
    *,
    speculative: bool,
) -> None:
    from app.audio import fail_safe_handoff, finalize_agent_response, send_agent_response

    try:
        if not should_process_transcript(transcript):
            logger.info("SKIP_NOISE_TURN [%s] transcript=%r", session.session_id, transcript)
            await log_turn_outcome(session, "no_response_needed")
            return
        session.interrupted = False
        session.response_buffer.start()
        result = await process_transcript(session, transcript)
        response_text = result.get("response_text") or result.get("response") or ""
        if session.twilio_websocket is not None and session.response_buffer.sentences_sent == 0 and response_text:
            await send_agent_response(
                websocket,
                session,
                response_text,
                transport="twilio",
                continue_response=True,
            )
            session.response_buffer.sentence_complete()
        if session.twilio_websocket is not None and session.response_buffer.sentences_sent > 0:
            await finalize_agent_response(websocket, session, transport="twilio")
        elif not response_text:
            await log_turn_outcome(session, "no_response_needed")
        session.pending_transcript = None
        if speculative:
            session.speculative_transcript = transcript
            session.speculative_task = asyncio.current_task()
        if session.should_close:
            await finalize_call(session, resolved=session.resolved)
            await websocket.close(code=1000)
    except Exception as exc:
        logger.error("TURN_ERROR [%s] %s", session.session_id, exc)
        await log_turn_outcome(session, "error")
        await fail_safe_handoff(websocket, session, "system_error", transport="twilio")


async def interrupt_twilio_playback(websocket: WebSocket, session: CallState) -> None:
    from app.audio import send_clear_to_twilio

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
