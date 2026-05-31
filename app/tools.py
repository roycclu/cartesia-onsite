from __future__ import annotations

import re
import logging
from typing import Any

from app.compliance import log_event, utc_now_iso
from mock_data.db import execute, fetch_all, fetch_one

logger = logging.getLogger("voice_agent")

NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "o": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def _spoken_digits_to_string(text: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    digits: list[str] = []
    for token in tokens:
        if token in NUMBER_WORDS:
            digits.append(NUMBER_WORDS[token])
        elif token.isdigit():
            digits.append(token)
    return "".join(digits)


def normalize_policy_number(value: str) -> str:
    cleaned = value.lower().replace("-", " ").replace(".", " ")
    tokens = re.findall(r"[a-z0-9]+", cleaned)
    normalized: list[str] = []
    for token in tokens:
        if token in NUMBER_WORDS:
            normalized.append(NUMBER_WORDS[token])
            continue
        if token == "policy":
            continue
        if token == "number":
            continue
        if token == "pol":
            normalized.append("POL")
            continue
        if all(part in NUMBER_WORDS for part in token.split()):
            normalized.extend(NUMBER_WORDS[part] for part in token.split())
            continue
        normalized.append(token.upper())
    collapsed = "".join(normalized)
    if not collapsed:
        return ""
    digits = "".join(char for char in collapsed if char.isdigit())
    if collapsed.startswith("POL"):
        return f"POL{digits}" if digits else "POL"
    if digits:
        return f"POL{digits}"
    return collapsed


def normalize_ssn_last4(value: str) -> str:
    digits = _spoken_digits_to_string(value)
    return digits[-4:] if len(digits) >= 4 else digits


async def resolve_policy_number(policy_number: str) -> str:
    normalized_input = normalize_policy_number(policy_number)
    if not normalized_input:
        return policy_number
    candidates = await fetch_all("SELECT policy_number FROM policies UNION SELECT policy_number FROM verification")
    for row in candidates:
        candidate = row["policy_number"]
        if normalize_policy_number(candidate) == normalized_input:
            return candidate
    return normalized_input


async def verify_identity(session_id: str, policy_number: str, ssn_last4: str) -> dict[str, Any]:
    resolved_policy_number = await resolve_policy_number(policy_number)
    normalized_ssn_last4 = normalize_ssn_last4(ssn_last4)
    logger.info(
        "verification_lookup session_id=%s policy_number=%s ssn_last4=%s",
        session_id,
        resolved_policy_number,
        normalized_ssn_last4,
    )
    record = await fetch_one(
        "SELECT policy_number, ssn_last4, holder_name FROM verification WHERE policy_number = $1",
        (resolved_policy_number,),
    )
    result = {
        "verified": bool(record and record["ssn_last4"] == normalized_ssn_last4),
        "policy_number": resolved_policy_number,
        "holder_name": record["holder_name"] if record else None,
    }
    logger.info(
        "TURN [%s] TOOL_CALL: verify_identity | INPUT: %s | OUTPUT: %s",
        session_id,
        {"policy_number": resolved_policy_number, "ssn_last4": normalized_ssn_last4},
        result,
    )
    await log_event(session_id, "identity_verification", result)
    return result


async def get_claim_status(session_id: str, policy_number: str) -> dict[str, Any]:
    resolved_policy_number = await resolve_policy_number(policy_number)
    claim = await fetch_one(
        "SELECT claim_id, policy_number, status, last_updated, adjuster_name FROM claims WHERE policy_number = $1",
        (resolved_policy_number,),
    )
    logger.info(
        "TURN [%s] TOOL_CALL: get_claim_status | INPUT: %s | OUTPUT: %s",
        session_id,
        {"policy_number": resolved_policy_number},
        claim,
    )
    await log_event(
        session_id,
        "tool_call",
        {"tool": "get_claim_status", "policy_number": resolved_policy_number, "result": claim},
    )
    return claim or {"error": "No claim found for that policy number."}


async def get_policy_info(session_id: str, policy_number: str) -> dict[str, Any]:
    resolved_policy_number = await resolve_policy_number(policy_number)
    policy = await fetch_one(
        "SELECT policy_number, holder_name, coverage_type, coverage_limit, deductible, effective_date FROM policies WHERE policy_number = $1",
        (resolved_policy_number,),
    )
    logger.info(
        "TURN [%s] TOOL_CALL: get_policy_info | INPUT: %s | OUTPUT: %s",
        session_id,
        {"policy_number": resolved_policy_number},
        policy,
    )
    await log_event(
        session_id,
        "tool_call",
        {"tool": "get_policy_info", "policy_number": resolved_policy_number, "result": policy},
    )
    return policy or {"error": "No policy found for that policy number."}


async def trigger_handoff(session_id: str, reason: str, transcript_summary: str) -> dict[str, Any]:
    payload = {
        "session_id": session_id,
        "reason_code": reason,
        "transcript_summary": transcript_summary,
        "timestamp": utc_now_iso(),
    }
    logger.info("TURN [%s] HANDOFF: %s", session_id, reason)
    await execute(
        "INSERT INTO handoff_queue(session_id, reason_code, transcript_summary, timestamp) VALUES ($1, $2, $3, $4)",
        (payload["session_id"], payload["reason_code"], payload["transcript_summary"], payload["timestamp"]),
    )
    await log_event(session_id, "handoff", payload)
    return payload
