#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import os

import asyncpg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report canonical turn latency from compliance logs.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours. Defaults to 24.")
    parser.add_argument("--session-id", help="Optional session_id filter for debugging a single call.")
    return parser.parse_args()


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def fmt_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0f}ms"


async def fetch_latency_rows(conn: asyncpg.Connection, hours: int, session_id: str | None) -> list[asyncpg.Record]:
    if session_id:
        return await conn.fetch(
            """
            WITH turn_latency AS (
              SELECT
                a.session_id,
                a.content::json->>'turn_id' AS turn_id,
                a.timestamp::timestamptz AS turn_end_ts,
                r.timestamp::timestamptz AS response_started_ts,
                (r.content::json->>'latency_ms')::int AS latency_ms,
                o.content::json->>'outcome' AS turn_outcome
              FROM compliance_log a
              JOIN compliance_log r
                ON a.session_id = r.session_id
               AND a.content::json->>'turn_id' = r.content::json->>'turn_id'
              LEFT JOIN compliance_log o
                ON a.session_id = o.session_id
               AND a.content::json->>'turn_id' = o.content::json->>'turn_id'
               AND o.event_type = 'turn_outcome'
              WHERE a.event_type = 'asr_result'
                AND r.event_type = 'response_started'
                AND a.timestamp::timestamptz >= NOW() - make_interval(hours => $1::int)
                AND a.session_id = $2
            )
            SELECT session_id, turn_id, turn_end_ts, response_started_ts, latency_ms, turn_outcome
            FROM turn_latency
            ORDER BY response_started_ts DESC
            """,
            hours,
            session_id,
        )
    return await conn.fetch(
        """
        WITH turn_latency AS (
          SELECT
            a.session_id,
            a.content::json->>'turn_id' AS turn_id,
            a.timestamp::timestamptz AS turn_end_ts,
            r.timestamp::timestamptz AS response_started_ts,
            (r.content::json->>'latency_ms')::int AS latency_ms,
            o.content::json->>'outcome' AS turn_outcome
          FROM compliance_log a
          JOIN compliance_log r
            ON a.session_id = r.session_id
           AND a.content::json->>'turn_id' = r.content::json->>'turn_id'
          LEFT JOIN compliance_log o
            ON a.session_id = o.session_id
           AND a.content::json->>'turn_id' = o.content::json->>'turn_id'
           AND o.event_type = 'turn_outcome'
          WHERE a.event_type = 'asr_result'
            AND r.event_type = 'response_started'
            AND a.timestamp::timestamptz >= NOW() - make_interval(hours => $1::int)
        )
        SELECT session_id, turn_id, turn_end_ts, response_started_ts, latency_ms, turn_outcome
        FROM turn_latency
        ORDER BY response_started_ts DESC
        """,
        hours,
    )


async def fetch_call_rows(conn: asyncpg.Connection, hours: int, session_id: str | None) -> list[asyncpg.Record]:
    if session_id:
        return await conn.fetch(
            """
            SELECT call_id, resolved, handoff_reason
            FROM calls
            WHERE started_at >= NOW() - make_interval(hours => $1::int)
              AND session_id::text = $2
            ORDER BY started_at DESC
            """,
            hours,
            session_id,
        )
    return await conn.fetch(
        """
        SELECT call_id, resolved, handoff_reason
        FROM calls
        WHERE started_at >= NOW() - make_interval(hours => $1::int)
        ORDER BY started_at DESC
        """,
        hours,
    )


async def main() -> None:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    conn = await asyncpg.connect(database_url)
    try:
        latency_rows = await fetch_latency_rows(conn, args.hours, args.session_id)
        call_rows = await fetch_call_rows(conn, args.hours, args.session_id)
    finally:
        await conn.close()

    outcome_counts = Counter((row["turn_outcome"] or "unknown") for row in latency_rows)
    responded_latencies = [row["latency_ms"] for row in latency_rows if row["turn_outcome"] == "responded" and row["latency_ms"] is not None]

    p50 = percentile(responded_latencies, 0.50)
    p75 = percentile(responded_latencies, 0.75)
    p95 = percentile(responded_latencies, 0.95)

    total_calls = len(call_rows)
    contained_calls = sum(1 for row in call_rows if row["resolved"] and not row["handoff_reason"])

    print("Latency Report")
    print("==============")
    print()
    print(f"Lookback window: last {args.hours} hour(s)")
    if args.session_id:
        print(f"Session filter: {args.session_id}")
    print()
    print("1. Turn End To First TTS Stream Start")
    print(f"p50: {fmt_ms(p50)}")
    print(f"p75: {fmt_ms(p75)}")
    print(f"p95: {fmt_ms(p95)}")
    print(f"Measured turns: {len(responded_latencies)}")
    print()
    print("2. Turn Outcomes")
    print(f"responded: {outcome_counts.get('responded', 0)}")
    print(f"no_response_needed: {outcome_counts.get('no_response_needed', 0)}")
    print(f"handoff: {outcome_counts.get('handoff', 0)}")
    print(f"error: {outcome_counts.get('error', 0)}")
    print(f"superseded: {outcome_counts.get('superseded', 0)}")
    print(f"unknown: {outcome_counts.get('unknown', 0)}")
    print()
    print("3. Call Summary")
    print(f"Total calls: {total_calls}")
    print(f"Contained calls: {contained_calls}")
    print()
    print("Sanity Check")
    print("------------")
    print(f"Joined latency rows: {len(latency_rows)}")
    print(f"Rows contributing to percentiles: {len(responded_latencies)}")
    print()
    print("Warnings")
    print("--------")
    has_warning = False
    if p50 is not None and p50 > 3000:
        print(f"WARNING: latency p50 is high at {fmt_ms(p50)}")
        has_warning = True
    if not responded_latencies:
        print("WARNING: no responded turns found in the selected window.")
        has_warning = True
    if not has_warning:
        print("No threshold warnings.")


if __name__ == "__main__":
    asyncio.run(main())
