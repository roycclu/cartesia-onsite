from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.db import dump_json, execute, fetch_all


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def log_event(session_id: str, event_type: str, content: Any) -> None:
    await execute(
        "INSERT INTO compliance_log(session_id, event_type, content, timestamp) VALUES ($1, $2, $3, $4)",
        (session_id, event_type, dump_json(content), utc_now_iso()),
    )


async def get_session_log(session_id: str) -> list[dict[str, Any]]:
    return await fetch_all(
        "SELECT session_id, event_type, content, timestamp FROM compliance_log WHERE session_id = $1 ORDER BY id ASC",
        (session_id,),
    )
