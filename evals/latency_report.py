#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any

import asyncpg


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}ms"


def fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}%"


async def fetch_rows(conn: asyncpg.Connection, query: str) -> list[asyncpg.Record]:
    return await conn.fetch(query)


def build_session_events(rows: list[asyncpg.Record]) -> dict[str, list[dict[str, Any]]]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        payload = json.loads(row["content"])
        event = {
            "event_type": row["event_type"],
            "timestamp": parse_ts(row["timestamp"]),
            "content": payload,
        }
        sessions.setdefault(row["session_id"], []).append(event)
    for events in sessions.values():
        events.sort(key=lambda event: event["timestamp"])
    return sessions


def analyze_sessions(sessions: dict[str, list[dict[str, Any]]]) -> tuple[list[float], list[float], int, int]:
    perceived_latency_ms: list[float] = []
    llm_response_ms: list[float] = []
    speculative_hits = 0
    speculative_total = 0

    for events in sessions.values():
        pending_eager_end: datetime | None = None
        pending_asr_result: datetime | None = None
        awaiting_speculative = False

        for event in events:
            event_type = event["event_type"]
            timestamp = event["timestamp"]

            if event_type == "turn_eager_end":
                transcript = (event["content"].get("transcript") or "").strip()
                if transcript:
                    pending_eager_end = timestamp
                    awaiting_speculative = True
                continue

            if event_type == "asr_result":
                pending_asr_result = timestamp
                if awaiting_speculative:
                    speculative_total += 1
                continue

            if event_type != "llm_response":
                continue

            if pending_eager_end is not None:
                perceived_latency_ms.append((timestamp - pending_eager_end).total_seconds() * 1000)
                if awaiting_speculative and pending_asr_result is None:
                    speculative_hits += 1
                pending_eager_end = None
                awaiting_speculative = False

            if pending_asr_result is not None:
                llm_response_ms.append((timestamp - pending_asr_result).total_seconds() * 1000)
                pending_asr_result = None

    return perceived_latency_ms, llm_response_ms, speculative_hits, speculative_total


async def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    conn = await asyncpg.connect(database_url)
    try:
        log_rows = await fetch_rows(
            conn,
            """
            SELECT session_id, event_type, content, timestamp
            FROM compliance_log
            WHERE event_type IN ('turn_eager_end', 'asr_result', 'llm_response')
            ORDER BY id ASC
            """,
        )
        call_rows = await fetch_rows(
            conn,
            """
            SELECT call_id, resolved, handoff_reason, duration_seconds, verified, turn_count
            FROM calls
            ORDER BY started_at DESC
            """,
        )
    finally:
        await conn.close()

    sessions = build_session_events(log_rows)
    perceived_latency_ms, llm_response_ms, speculative_hits, speculative_total = analyze_sessions(sessions)

    perceived_p50 = percentile(perceived_latency_ms, 0.50)
    perceived_p75 = percentile(perceived_latency_ms, 0.75)
    perceived_p95 = percentile(perceived_latency_ms, 0.95)
    llm_p50 = percentile(llm_response_ms, 0.50)
    llm_p95 = percentile(llm_response_ms, 0.95)
    speculative_hit_rate = (speculative_hits / speculative_total * 100) if speculative_total else None

    total_calls = len(call_rows)
    contained_calls = sum(1 for row in call_rows if row["resolved"] and not row["handoff_reason"])
    containment_rate = (contained_calls / total_calls * 100) if total_calls else None

    print("Latency Report")
    print("==============")
    print()
    print("1. Perceived Latency")
    print(f"p50: {fmt_ms(perceived_p50)}")
    print(f"p75: {fmt_ms(perceived_p75)}")
    print(f"p95: {fmt_ms(perceived_p95)}")
    print()
    print("2. LLM Response Time")
    print(f"p50: {fmt_ms(llm_p50)}")
    print(f"p95: {fmt_ms(llm_p95)}")
    print()
    print("3. Speculative Execution")
    print(f"Hit rate: {fmt_pct(speculative_hit_rate)}")
    print(f"Measured turns: {speculative_total}")
    print()
    print("4. Call Summary")
    print(f"Total calls: {total_calls}")
    print(f"Contained calls: {contained_calls}")
    print(f"Containment rate: {fmt_pct(containment_rate)}")
    print()
    print("Warnings")
    print("--------")

    has_warning = False
    if perceived_p95 is not None and perceived_p95 > 500:
        print(f"WARNING: perceived latency p95 is high at {fmt_ms(perceived_p95)}")
        has_warning = True
    if speculative_hit_rate is not None and speculative_hit_rate < 80:
        print(f"WARNING: speculative hit rate is low at {fmt_pct(speculative_hit_rate)}")
        has_warning = True
    if containment_rate is not None and containment_rate < 60:
        print(f"WARNING: containment rate is low at {fmt_pct(containment_rate)}")
        has_warning = True
    if not has_warning:
        print("No threshold warnings.")


if __name__ == "__main__":
    asyncio.run(main())
