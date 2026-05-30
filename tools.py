from __future__ import annotations

from typing import Any

from compliance import log_event, utc_now_iso
from db import execute, fetch_one


async def verify_identity(session_id: str, policy_number: str, ssn_last4: str) -> dict[str, Any]:
    record = await fetch_one(
        "SELECT policy_number, ssn_last4, holder_name FROM verification WHERE policy_number = ?",
        (policy_number,),
    )
    result = {
        "verified": bool(record and record["ssn_last4"] == ssn_last4),
        "policy_number": policy_number,
        "holder_name": record["holder_name"] if record else None,
    }
    await log_event(session_id, "identity_verification", result)
    return result


async def get_claim_status(session_id: str, policy_number: str) -> dict[str, Any]:
    claim = await fetch_one(
        "SELECT claim_id, policy_number, status, last_updated, adjuster_name FROM claims WHERE policy_number = ?",
        (policy_number,),
    )
    await log_event(session_id, "tool_call", {"tool": "get_claim_status", "policy_number": policy_number, "result": claim})
    return claim or {"error": "No claim found for that policy number."}


async def get_policy_info(session_id: str, policy_number: str) -> dict[str, Any]:
    policy = await fetch_one(
        "SELECT policy_number, holder_name, coverage_type, coverage_limit, deductible, effective_date FROM policies WHERE policy_number = ?",
        (policy_number,),
    )
    await log_event(session_id, "tool_call", {"tool": "get_policy_info", "policy_number": policy_number, "result": policy})
    return policy or {"error": "No policy found for that policy number."}


async def trigger_handoff(session_id: str, reason: str, transcript_summary: str) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "reason_code": reason,
        "transcript_summary": transcript_summary,
        "timestamp": utc_now_iso(),
    }
    await execute(
        "INSERT INTO handoff_queue(session_id, reason_code, transcript_summary, timestamp) VALUES (?, ?, ?, ?)",
        (payload["session_id"], payload["reason_code"], payload["transcript_summary"], payload["timestamp"]),
    )
    await log_event(session_id, "handoff", payload)
    return payload
