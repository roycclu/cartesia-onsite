from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

from app.pii import mask_policy
from app.response_buffer import ResponseBuffer

logger = logging.getLogger("voice_agent")


@dataclass
class ConversationState:
    verified: bool = False
    policy_number: Optional[str] = None
    ssn_last4: Optional[str] = None
    holder_name: Optional[str] = None
    verification_attempts: int = 0
    pending_intent: Optional[str] = None
    pending_intent_transcript: Optional[str] = None
    answered_queries: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_tool_result: dict[str, Any] | None = None
    should_handoff: bool = False
    handoff_reason: Optional[str] = None
    should_close: bool = False
    resolved: bool = False
    name_acknowledged: bool = False
    history: list[dict[str, str]] = field(default_factory=list)
    turn_count: int = 0
    human_requests: int = 0


@dataclass
class SessionRuntime:
    session_id: str | None = None
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    call_sid: str | None = None
    stream_sid: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: Optional[datetime] = None
    twilio_websocket: Any = field(default=None, repr=False)
    twilio_send_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    ink_stream: Any = field(default=None, repr=False)
    active_response_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    active_tts_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    speculative_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    speculative_transcript: str | None = None
    tts_playing: bool = False
    last_mark: str | None = None
    interrupted: bool = False
    pending_transcript: str | None = None
    current_tts_context_id: str | None = None
    current_tts_ws: Any = field(default=None, repr=False)
    current_tts_reader_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    current_tts_open: bool = False
    current_tts_finalized: bool = False
    current_tts_has_input: bool = False
    response_buffer: ResponseBuffer = field(default_factory=ResponseBuffer, repr=False)
    current_turn_id: str | None = None
    current_turn_end_latency_t0: float | None = None
    current_turn_response_started_logged: bool = False
    current_turn_outcome: str | None = None


@dataclass
class CallState:
    conversation: ConversationState = field(default_factory=ConversationState)
    runtime: SessionRuntime = field(default_factory=SessionRuntime)
    prompt_version: str | None = None
    call_summary: Optional[dict[str, Any]] = None
    eval_pii_safety: int | None = None
    eval_intent_acknowledgment: int | None = None

    _CONVERSATION_FIELDS = {
        "verified",
        "policy_number",
        "ssn_last4",
        "holder_name",
        "verification_attempts",
        "pending_intent",
        "pending_intent_transcript",
        "answered_queries",
        "latest_tool_result",
        "should_handoff",
        "handoff_reason",
        "should_close",
        "resolved",
        "name_acknowledged",
        "history",
        "turn_count",
        "human_requests",
    }
    _RUNTIME_FIELDS = {
        "session_id",
        "call_id",
        "call_sid",
        "stream_sid",
        "started_at",
        "ended_at",
        "twilio_websocket",
        "twilio_send_lock",
        "ink_stream",
        "active_response_task",
        "active_tts_task",
        "speculative_task",
        "speculative_transcript",
        "tts_playing",
        "last_mark",
        "interrupted",
        "pending_transcript",
        "current_tts_context_id",
        "current_tts_ws",
        "current_tts_reader_task",
        "current_tts_open",
        "current_tts_finalized",
        "current_tts_has_input",
        "response_buffer",
        "current_turn_id",
        "current_turn_end_latency_t0",
        "current_turn_response_started_logged",
        "current_turn_outcome",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self._CONVERSATION_FIELDS:
            return getattr(self.conversation, name)
        if name in self._RUNTIME_FIELDS:
            return getattr(self.runtime, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"conversation", "runtime", "prompt_version", "call_summary", "eval_pii_safety", "eval_intent_acknowledgment", "_CONVERSATION_FIELDS", "_RUNTIME_FIELDS"}:
            object.__setattr__(self, name, value)
            return
        if "conversation" in self.__dict__ and name in self._CONVERSATION_FIELDS:
            setattr(self.conversation, name, value)
            return
        if "runtime" in self.__dict__ and name in self._RUNTIME_FIELDS:
            setattr(self.runtime, name, value)
            return
        object.__setattr__(self, name, value)

    def merge_extracted_fields(self, extracted: dict[str, Any]) -> None:
        if extracted.get("policy_number"):
            self.policy_number = extracted["policy_number"]
        if extracted.get("ssn_last4"):
            self.ssn_last4 = extracted["ssn_last4"]

    def capture_pending_intent(self, intent: str, transcript: str) -> None:
        if not self.verified and intent not in ("verify_identity", "unknown"):
            self.pending_intent = intent
            self.pending_intent_transcript = transcript

    def start_new_turn(self) -> str:
        self.current_turn_id = str(uuid.uuid4())[:8]
        self.current_turn_end_latency_t0 = None
        self.current_turn_response_started_logged = False
        self.current_turn_outcome = None
        return self.current_turn_id

    def record_answered_query(self, query_type: str, result: dict[str, Any]) -> None:
        self.answered_queries[query_type] = result
        self.latest_tool_result = result

    def mark_verification_failed(self) -> None:
        self.ssn_last4 = None
        self.verification_attempts += 1

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
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": (self.ended_at - self.started_at).total_seconds(),
            "verified": self.verified,
            "policy_number": mask_policy(self.policy_number),
            "turn_count": self.turn_count,
            "resolved": self.resolved,
            "handoff_reason": self.handoff_reason,
            "answered_queries": list(self.answered_queries.keys()),
            "prompt_version": self.prompt_version,
            "eval_pii_safety": self.eval_pii_safety,
            "eval_intent_acknowledgment": self.eval_intent_acknowledgment,
        }

    def to_llm_state(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "policy_number": self.policy_number,
            "holder_name": self.holder_name,
            "name_acknowledged": self.name_acknowledged,
            "turn_count": self.turn_count,
            "pending_intent": self.pending_intent,
            "answered_queries": self.answered_queries,
            "latest_tool_result": self.latest_tool_result,
            "recent_history": self.history[-4:] if self.history else [],
            "should_handoff": self.should_handoff,
            "handoff_reason": self.handoff_reason,
        }
