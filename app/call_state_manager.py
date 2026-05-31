from __future__ import annotations

import logging
from typing import Optional

from app.call_state import CallState
from app.pii import mask_policy

logger = logging.getLogger("voice_agent")


class CallStateManager:
    def __init__(self) -> None:
        self._active_calls: dict[str, CallState] = {}

    def create(self, session_id: str, call_sid: str | None) -> CallState:
        state = CallState(session_id=session_id, call_sid=call_sid)
        self._active_calls[session_id] = state
        logger.info("CALL_START call_id=%s session_id=%s", state.call_id, session_id)
        return state

    def get(self, session_id: str) -> Optional[CallState]:
        return self._active_calls.get(session_id)

    def end(self, session_id: str, resolved: bool = False) -> Optional[CallState]:
        state = self._active_calls.pop(session_id, None)
        if state:
            state.end_call(resolved=resolved)
            assert state.call_summary is not None
            logger.info(
                "CALL_END call_id=%s duration=%.1fs turns=%s resolved=%s handoff=%s policy=%s",
                state.call_id,
                state.call_summary["duration_seconds"],
                state.turn_count,
                resolved,
                state.handoff_reason,
                mask_policy(state.policy_number),
            )
        return state

    def active_count(self) -> int:
        return len(self._active_calls)


call_state_manager = CallStateManager()
