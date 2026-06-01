#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg


@dataclass
class TurnMetrics:
    turn_end: datetime | None = None
    response_started: datetime | None = None
    turn_outcome: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report aggregate latency metrics from compliance logs.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours. Defaults to 24.")
    return parser.parse_args()


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


async def fetch_rows(conn: asyncpg.Connection, query: str, *params: Any) -> list[asyncpg.Record]:
    return await conn.fetch(query, *params)


def build_turn_metrics(rows: list[asyncpg.Record]) -> dict[str, TurnMetrics]:
    turns: dict[str, TurnMetrics] = {}
    for row in rows:
        payload = json.loads(row["content"])
        turn_id = payload.get("turn_id")
        if not turn_id:
            continue
        turn = turns.setdefault(turn_id, TurnMetrics())
        timestamp = parse_ts(row["timestamp"])
        if row["event_type"] == "asr_result":
            turn.turn_end = timestamp
        elif row["event_type"] == "response_started":
            if turn.response_started is None or timestamp < turn.response_started:
                turn.response_started = timestamp
        elif row["event_type"] == "turn_outcome":
            turn.turn_outcome = payload.get("outcome")
    return turns


def analyze_turns(turns: dict[str, TurnMetrics]) -> tuple[list[float], dict[str, int]]:
    latency_ms: list[float] = []
    outcome_counts: dict[str, int] = {}

    for turn in turns.values():
        outcome = turn.turn_outcome or "unknown"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if outcome != "responded":
            continue
        if turn.turn_end is None or turn.response_started is None:
            continue
        latency_ms.append(max(0.0, (turn.response_started - turn.turn_end).total_seconds() * 1000))

    return latency_ms, outcome_counts


async def main() -> None:
    args = parse_args()
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
            WHERE event_type IN ('asr_result', 'response_started', 'turn_outcome')
              AND timestamp::timestamptz >= NOW() - make_interval(hours => $1::int)
            ORDER BY id ASC
            """,
            args.hours,
        )
        call_rows = await fetch_rows(
            conn,
            """
            SELECT call_id, resolved, handoff_reason, duration_seconds, verified, turn_count
            FROM calls
            WHERE started_at >= NOW() - make_interval(hours => $1::int)
            ORDER BY started_at DESC
            """,
            args.hours,
        )
    finally:
        await conn.close()

    turns = build_turn_metrics(log_rows)
    latency_ms, outcome_counts = analyze_turns(turns)

    p50 = percentile(latency_ms, 0.50)
    p75 = percentile(latency_ms, 0.75)
    p95 = percentile(latency_ms, 0.95)

    total_calls = len(call_rows)
    contained_calls = sum(1 for row in call_rows if row["resolved"] and not row["handoff_reason"])
    containment_rate = (contained_calls / total_calls * 100) if total_calls else None

    print("Latency Report")
    print("==============")
    print()
    print(f"Lookback window: last {args.hours} hour(s)")
    print()
    print("1. Turn End To First TTS Stream Start")
    print(f"p50: {fmt_ms(p50)}")
    print(f"p75: {fmt_ms(p75)}")
    print(f"p95: {fmt_ms(p95)}")
    print(f"Measured turns: {len(latency_ms)}")
    print()
    print("2. Turn Outcomes")
    print(f"responded: {outcome_counts.get('responded', 0)}")
    print(f"no_response_needed: {outcome_counts.get('no_response_needed', 0)}")
    print(f"handoff: {outcome_counts.get('handoff', 0)}")
    print(f"error: {outcome_counts.get('error', 0)}")
    print(f"superseded: {outcome_counts.get('superseded', 0)}")
    print()
    print("3. Call Summary")
    print(f"Total calls: {total_calls}")
    print(f"Contained calls: {contained_calls}")
    print(f"Containment rate: {fmt_pct(containment_rate)}")
    print()
    print("Warnings")
    print("--------")

    has_warning = False
    if p50 is not None and p50 > 3000:
        print(f"WARNING: latency p50 is high at {fmt_ms(p50)}; rerun with --hours 2 to isolate recent calls.")
        has_warning = True
    if containment_rate is not None and containment_rate < 60:
        print(f"WARNING: containment rate is low at {fmt_pct(containment_rate)}")
        has_warning = True
    if not has_warning:
        print("No threshold warnings.")


if __name__ == "__main__":
    asyncio.run(main())
