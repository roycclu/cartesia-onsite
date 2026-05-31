from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


@dataclass
class CallState:
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    call_sid: str | None = None
    stream_sid: str | None = None

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None

    verified: bool = False
    policy_number: Optional[str] = None
    ssn_last4: Optional[str] = None
    verification_attempts: int = 0

    history: list[dict[str, str]] = field(default_factory=list)
    pending_intent: Optional[str] = None
    answered_queries: dict[str, dict[str, Any]] = field(default_factory=dict)
    turn_count: int = 0

    should_handoff: bool = False
    handoff_reason: Optional[str] = None
    should_close: bool = False
    resolved: bool = False

    prompt_version: str | None = None
    call_summary: Optional[dict[str, Any]] = None

    twilio_send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    tts_playing: bool = False
    active_response_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    active_tts_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    pending_transcript: str | None = None
    ink_stream: Any = field(default=None, repr=False)

    def merge_extracted_fields(self, extracted: dict[str, Any]) -> None:
        if extracted.get("policy_number") and not self.policy_number:
            self.policy_number = extracted["policy_number"]
        if extracted.get("ssn_last4") and not self.ssn_last4:
            self.ssn_last4 = extracted["ssn_last4"]

    def capture_pending_intent(self, intent: str) -> None:
        if not self.verified and intent not in ("verify_identity", "unknown"):
            self.pending_intent = intent

    def record_answered_query(self, query_type: str, result: dict[str, Any]) -> None:
        self.answered_queries[query_type] = result

    def already_answered(self, query_type: str) -> bool:
        return query_type in self.answered_queries

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.turn_count += 1

    def end_call(self, resolved: bool = False) -> None:
        self.ended_at = datetime.now(timezone.utc)
        self.resolved = resolved
        self.call_summary = {
            "call_id": self.call_id,
            "call_sid": self.call_sid,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "duration_seconds": (self.ended_at - self.started_at).total_seconds(),
            "verified": self.verified,
            "policy_number": self.policy_number,
            "turn_count": self.turn_count,
            "resolved": self.resolved,
            "handoff_reason": self.handoff_reason,
            "answered_queries": list(self.answered_queries.keys()),
            "prompt_version": self.prompt_version,
        }

    def to_llm_state(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "policy_number": self.policy_number,
            "turn_count": self.turn_count,
            "pending_intent": self.pending_intent,
            "answered_queries": list(self.answered_queries.keys()),
            "should_handoff": self.should_handoff,
            "handoff_reason": self.handoff_reason,
        }
