from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.call_state import CallState
from app.pii import hash_policy, mask_transcript
from app.prompts import PROMPT_VERSION
from mock_data.db import dump_json, execute, fetch_all


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def log_event(session_id: str, event_type: str, content: Any) -> None:
    if isinstance(content, dict):
        payload = dict(content)
        payload.setdefault("prompt_version", PROMPT_VERSION)
    else:
        payload = {"value": content, "prompt_version": PROMPT_VERSION}
    if event_type == "user_transcript":
        payload["text"] = mask_transcript(payload.get("text"))
    await execute(
        "INSERT INTO compliance_log(session_id, event_type, content, timestamp) VALUES ($1, $2, $3, $4)",
        (session_id, event_type, dump_json(payload), utc_now_iso()),
    )


async def get_session_log(session_id: str) -> list[dict[str, Any]]:
    return await fetch_all(
        "SELECT session_id, event_type, content, timestamp FROM compliance_log WHERE session_id = $1 ORDER BY id ASC",
        (session_id,),
    )


async def persist_call_record(state: CallState, db_pool: Any, *, log_finalized_event: bool = True) -> None:
    if state.call_summary is None:
        state.end_call(resolved=state.resolved)
    assert state.call_summary is not None
    state.call_summary["eval_pii_safety"] = state.eval_pii_safety
    state.call_summary["eval_intent_acknowledgment"] = state.eval_intent_acknowledgment

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO calls (
                call_id, call_sid, session_id, started_at, ended_at,
                duration_seconds, verified, policy_number, turn_count,
                resolved, handoff_reason, answered_queries, prompt_version,
                eval_pii_safety, eval_intent_acknowledgment
            ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
            ON CONFLICT (call_id) DO UPDATE SET
                call_sid = EXCLUDED.call_sid,
                session_id = EXCLUDED.session_id,
                started_at = EXCLUDED.started_at,
                ended_at = EXCLUDED.ended_at,
                duration_seconds = EXCLUDED.duration_seconds,
                verified = EXCLUDED.verified,
                policy_number = EXCLUDED.policy_number,
                turn_count = EXCLUDED.turn_count,
                resolved = EXCLUDED.resolved,
                handoff_reason = EXCLUDED.handoff_reason,
                answered_queries = EXCLUDED.answered_queries,
                prompt_version = EXCLUDED.prompt_version,
                eval_pii_safety = EXCLUDED.eval_pii_safety,
                eval_intent_acknowledgment = EXCLUDED.eval_intent_acknowledgment
            """,
            state.call_summary["call_id"],
            state.call_summary["call_sid"],
            state.call_summary["session_id"],
            state.call_summary["started_at"],
            state.call_summary["ended_at"],
            state.call_summary["duration_seconds"],
            state.call_summary["verified"],
            hash_policy(state.policy_number),
            state.call_summary["turn_count"],
            state.call_summary["resolved"],
            state.call_summary["handoff_reason"],
            str(state.call_summary["answered_queries"]),
            state.call_summary["prompt_version"],
            state.call_summary["eval_pii_safety"],
            state.call_summary["eval_intent_acknowledgment"],
        )
        await conn.execute("DELETE FROM call_turns WHERE call_id = $1", state.call_id)
        for i, turn in enumerate(state.history):
            await conn.execute(
                """
                INSERT INTO call_turns (call_id, turn_index, role, content)
                VALUES ($1, $2, $3, $4)
                """,
                state.call_id,
                i,
                turn["role"],
                turn["content"],
            )

    if log_finalized_event:
        await log_event(state.session_id or state.call_id, "call_finalized", state.call_summary)
