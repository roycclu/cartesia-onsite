#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from typing import Any

import asyncpg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw latency-related rows from compliance logs.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours. Defaults to 24.")
    parser.add_argument("--session-id", help="Optional session_id filter.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to return. Defaults to 100.")
    parser.add_argument(
        "--view",
        choices=("joined", "response_started", "asr_result", "turn_outcome", "latency"),
        default="joined",
        help="Which raw view to fetch. Defaults to joined.",
    )
    parser.add_argument(
        "--format",
        choices=("json", "table", "csv"),
        default="json",
        help="Output format. Defaults to json.",
    )
    return parser.parse_args()


def _json_default(value: Any) -> str:
    return str(value)


def build_query(view: str, has_session_id: bool) -> str:
    session_clause = "AND session_id = $3" if has_session_id else ""
    if view == "joined":
        return f"""
            SELECT
              a.session_id,
              a.content::json->>'turn_id' AS turn_id,
              a.timestamp::timestamptz AS asr_result_ts,
              r.timestamp::timestamptz AS response_started_ts,
              (r.content::json->>'latency_ms')::int AS latency_ms,
              o.content::json->>'outcome' AS turn_outcome,
              a.content::json AS asr_result_content,
              r.content::json AS response_started_content,
              o.content::json AS turn_outcome_content
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
              {session_clause}
            ORDER BY r.timestamp::timestamptz DESC
            LIMIT $2
        """
    return f"""
        SELECT session_id, event_type, timestamp::timestamptz AS event_ts, content::json AS content
        FROM compliance_log
        WHERE event_type = '{view}'
          AND timestamp::timestamptz >= NOW() - make_interval(hours => $1::int)
          {session_clause}
        ORDER BY timestamp::timestamptz DESC
        LIMIT $2
    """


async def fetch_rows(conn: asyncpg.Connection, args: argparse.Namespace) -> list[asyncpg.Record]:
    query = build_query(args.view, bool(args.session_id))
    if args.session_id:
        return await conn.fetch(query, args.hours, args.limit, args.session_id)
    return await conn.fetch(query, args.hours, args.limit)


def normalize_row(row: asyncpg.Record) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in dict(row).items():
        if isinstance(value, str):
            try:
                normalized[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        normalized[key] = value
    return normalized


def print_json(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(json.dumps(row, default=_json_default))


def print_csv(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: json.dumps(value, default=_json_default) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    rendered_rows = [
        [
            json.dumps(value, default=_json_default) if isinstance(value, (dict, list)) else str(value)
            for value in row.values()
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for rendered in rendered_rows:
        for i, value in enumerate(rendered):
            widths[i] = max(widths[i], len(value))
    print(" | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for rendered in rendered_rows:
        print(" | ".join(value.ljust(widths[i]) for i, value in enumerate(rendered)))


async def main() -> None:
    args = parse_args()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    conn = await asyncpg.connect(database_url)
    try:
        rows = await fetch_rows(conn, args)
    finally:
        await conn.close()

    normalized_rows = [normalize_row(row) for row in rows]
    if args.format == "json":
        print_json(normalized_rows)
    elif args.format == "csv":
        print_csv(normalized_rows)
    else:
        print_table(normalized_rows)


if __name__ == "__main__":
    asyncio.run(main())
