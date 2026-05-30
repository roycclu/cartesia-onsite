from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "insurance_demo.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    policy_number TEXT NOT NULL,
    status TEXT NOT NULL,
    last_updated TEXT NOT NULL,
    adjuster_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_number TEXT PRIMARY KEY,
    holder_name TEXT NOT NULL,
    coverage_type TEXT NOT NULL,
    coverage_limit INTEGER NOT NULL,
    deductible INTEGER NOT NULL,
    effective_date TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification (
    policy_number TEXT PRIMARY KEY,
    ssn_last4 TEXT NOT NULL,
    holder_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS handoff_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    reason_code TEXT NOT NULL,
    transcript_summary TEXT NOT NULL,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS compliance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
"""


SEED_POLICIES = [
    ("POL-1001", "Maya Patel", "Auto Premium", 100000, 500, "2025-01-01"),
    ("POL-1002", "Jordan Kim", "Home Plus", 450000, 1500, "2024-11-15"),
    ("POL-1003", "Avery Johnson", "Renters Basic", 50000, 1000, "2025-02-10"),
    ("POL-1004", "Sofia Ramirez", "Auto Standard", 75000, 750, "2024-09-20"),
    ("POL-1005", "Ethan Brooks", "Life Shield", 250000, 0, "2025-03-05"),
]

SEED_CLAIMS = [
    ("CLM-9001", "POL-1001", "Under review", "2026-05-28T14:15:00Z", "Dana Moore"),
    ("CLM-9002", "POL-1002", "Approved", "2026-05-27T18:45:00Z", "Chris Nguyen"),
    ("CLM-9003", "POL-1003", "Pending documents", "2026-05-24T09:30:00Z", "Taylor Reed"),
    ("CLM-9004", "POL-1004", "Closed", "2026-05-18T11:10:00Z", "Morgan Ellis"),
    ("CLM-9005", "POL-1005", "Payment issued", "2026-05-29T16:05:00Z", "Jamie Clark"),
]

SEED_VERIFICATION = [
    ("POL-1001", "4821", "Maya Patel"),
    ("POL-1002", "1934", "Jordan Kim"),
    ("POL-1003", "6407", "Avery Johnson"),
    ("POL-1004", "7752", "Sofia Ramirez"),
    ("POL-1005", "2219", "Ethan Brooks"),
]


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


async def init_db() -> None:
    _init_db_sync()


async def fetch_one(query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    return _fetch_one_sync(query, params)


async def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return _fetch_all_sync(query, params)


async def execute(query: str, params: tuple[Any, ...]) -> None:
    _execute_sync(query, params)


def dump_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def _init_db_sync() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        conn.execute("DELETE FROM claims")
        conn.execute("DELETE FROM policies")
        conn.execute("DELETE FROM verification")
        conn.executemany(
            "INSERT INTO policies(policy_number, holder_name, coverage_type, coverage_limit, deductible, effective_date) VALUES (?, ?, ?, ?, ?, ?)",
            SEED_POLICIES,
        )
        conn.executemany(
            "INSERT INTO claims(claim_id, policy_number, status, last_updated, adjuster_name) VALUES (?, ?, ?, ?, ?)",
            SEED_CLAIMS,
        )
        conn.executemany(
            "INSERT INTO verification(policy_number, ssn_last4, holder_name) VALUES (?, ?, ?)",
            SEED_VERIFICATION,
        )
        conn.commit()


def _fetch_one_sync(query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(query, params).fetchone()
        return dict(row) if row else None


def _fetch_all_sync(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def _execute_sync(query: str, params: tuple[Any, ...]) -> None:
    with get_connection() as conn:
        conn.execute(query, params)
        conn.commit()
