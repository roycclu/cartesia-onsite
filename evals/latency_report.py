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
    eager_end: datetime | None = None
    turn_end: datetime | None = None
    response_started: datetime | None = None
    speculative_outcome: str | None = None
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
        if row["event_type"] == "turn_eager_end":
            turn.eager_end = timestamp
        elif row["event_type"] == "asr_result":
            turn.turn_end = timestamp
        elif row["event_type"] == "response_started":
            if turn.response_started is None or timestamp < turn.response_started:
                turn.response_started = timestamp
        elif row["event_type"] == "speculative_resolution":
            turn.speculative_outcome = payload.get("outcome")
        elif row["event_type"] == "turn_outcome":
            turn.turn_outcome = payload.get("outcome")
    return turns


def analyze_turns(turns: dict[str, TurnMetrics]) -> tuple[list[float], list[float], int, int]:
    eager_end_to_response_started_ms: list[float] = []
    turn_end_to_response_started_ms: list[float] = []
    speculative_total = 0
    speculative_hits = 0

    for turn in turns.values():
        if turn.speculative_outcome is not None:
            speculative_total += 1
            if turn.speculative_outcome == "hit":
                speculative_hits += 1
        if turn.turn_outcome != "responded" or turn.speculative_outcome == "miss":
            continue
        if turn.eager_end is not None and turn.response_started is not None:
            eager_end_to_response_started_ms.append(max(0.0, (turn.response_started - turn.eager_end).total_seconds() * 1000))
        if turn.turn_end is not None and turn.response_started is not None:
            turn_end_to_response_started_ms.append(max(0.0, (turn.response_started - turn.turn_end).total_seconds() * 1000))

    return eager_end_to_response_started_ms, turn_end_to_response_started_ms, speculative_hits, speculative_total


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
            WHERE event_type IN ('turn_eager_end', 'asr_result', 'response_started', 'speculative_resolution', 'turn_outcome')
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
    eager_end_to_response_started_ms, turn_end_to_response_started_ms, speculative_hits, speculative_total = analyze_turns(turns)

    eager_p50 = percentile(eager_end_to_response_started_ms, 0.50)
    eager_p75 = percentile(eager_end_to_response_started_ms, 0.75)
    eager_p95 = percentile(eager_end_to_response_started_ms, 0.95)
    turn_end_p50 = percentile(turn_end_to_response_started_ms, 0.50)
    turn_end_p75 = percentile(turn_end_to_response_started_ms, 0.75)
    turn_end_p95 = percentile(turn_end_to_response_started_ms, 0.95)
    speculative_hit_rate = (speculative_hits / speculative_total * 100) if speculative_total else None

    total_calls = len(call_rows)
    contained_calls = sum(1 for row in call_rows if row["resolved"] and not row["handoff_reason"])
    containment_rate = (contained_calls / total_calls * 100) if total_calls else None

    print("Latency Report")
    print("==============")
    print()
    print(f"Lookback window: last {args.hours} hour(s)")
    print()
    print("1. Turn End To Response Started")
    print(f"p50: {fmt_ms(turn_end_p50)}")
    print(f"p75: {fmt_ms(turn_end_p75)}")
    print(f"p95: {fmt_ms(turn_end_p95)}")
    print()
    print("2. Eager End To Response Started")
    print(f"p50: {fmt_ms(eager_p50)}")
    print(f"p75: {fmt_ms(eager_p75)}")
    print(f"p95: {fmt_ms(eager_p95)}")
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
    if turn_end_p50 is not None and turn_end_p50 > 3000:
        print(f"WARNING: turn-end p50 is high at {fmt_ms(turn_end_p50)}; rerun with --hours 2 to isolate recent calls.")
        has_warning = True
    if eager_p50 is not None and eager_p50 > 3000:
        print(f"WARNING: eager-end p50 is high at {fmt_ms(eager_p50)}; rerun with --hours 2 to isolate recent calls.")
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
