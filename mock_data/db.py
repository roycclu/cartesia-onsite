from __future__ import annotations

import json
from typing import Any

import asyncpg

from app import config

pool: asyncpg.Pool | None = None


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        policy_number TEXT NOT NULL,
        status TEXT NOT NULL,
        last_updated TEXT NOT NULL,
        adjuster_name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS policies (
        policy_number TEXT PRIMARY KEY,
        holder_name TEXT NOT NULL,
        coverage_type TEXT NOT NULL,
        coverage_limit INTEGER NOT NULL,
        deductible INTEGER NOT NULL,
        effective_date TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS verification (
        policy_number TEXT PRIMARY KEY,
        ssn_last4 TEXT NOT NULL,
        holder_name TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS handoff_queue (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        reason_code TEXT NOT NULL,
        transcript_summary TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS compliance_log (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS calls (
        call_id UUID PRIMARY KEY,
        call_sid TEXT,
        session_id UUID,
        started_at TIMESTAMPTZ,
        ended_at TIMESTAMPTZ,
        duration_seconds FLOAT,
        verified BOOLEAN,
        policy_number TEXT,
        turn_count INTEGER,
        resolved BOOLEAN,
        handoff_reason TEXT,
        answered_queries TEXT,
        prompt_version TEXT
    )
    """,
    """
    ALTER TABLE calls
    ADD COLUMN IF NOT EXISTS eval_pii_safety INTEGER
    """,
    """
    ALTER TABLE calls
    ADD COLUMN IF NOT EXISTS eval_intent_acknowledgment INTEGER
    """,
    """
    CREATE TABLE IF NOT EXISTS call_turns (
        id SERIAL PRIMARY KEY,
        call_id UUID REFERENCES calls(call_id),
        turn_index INTEGER,
        role TEXT,
        content TEXT
    )
    """,
    """
    CREATE OR REPLACE FUNCTION prevent_compliance_log_mutation()
    RETURNS trigger AS $$
    BEGIN
        RAISE EXCEPTION 'compliance_log is append-only';
    END;
    $$ LANGUAGE plpgsql
    """,
    """
    DROP TRIGGER IF EXISTS compliance_log_no_update ON compliance_log
    """,
    """
    CREATE TRIGGER compliance_log_no_update
    BEFORE UPDATE ON compliance_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_compliance_log_mutation()
    """,
    """
    DROP TRIGGER IF EXISTS compliance_log_no_delete ON compliance_log
    """,
    """
    CREATE TRIGGER compliance_log_no_delete
    BEFORE DELETE ON compliance_log
    FOR EACH ROW
    EXECUTE FUNCTION prevent_compliance_log_mutation()
    """,
]


SEED_POLICIES = [
    ("POL1001", "Maya Patel", "Auto Premium", 100000, 500, "2025-01-01"),
    ("POL1002", "Jordan Kim", "Home Plus", 450000, 1500, "2024-11-15"),
    ("POL1003", "Avery Johnson", "Renters Basic", 50000, 1000, "2025-02-10"),
    ("POL1004", "Sofia Ramirez", "Auto Standard", 75000, 750, "2024-09-20"),
    ("POL1005", "Ethan Brooks", "Life Shield", 250000, 0, "2025-03-05"),
]

SEED_CLAIMS = [
    ("CLM-9001", "POL1001", "Under review", "2026-05-28T14:15:00Z", "Dana Moore"),
    ("CLM-9002", "POL1002", "Approved", "2026-05-27T18:45:00Z", "Chris Nguyen"),
    ("CLM-9003", "POL1003", "Pending documents", "2026-05-24T09:30:00Z", "Taylor Reed"),
    ("CLM-9004", "POL1004", "Closed", "2026-05-18T11:10:00Z", "Morgan Ellis"),
    ("CLM-9005", "POL1005", "Payment issued", "2026-05-29T16:05:00Z", "Jamie Clark"),
]

SEED_VERIFICATION = [
    ("POL1001", "4821", "Maya Patel"),
    ("POL1002", "1934", "Jordan Kim"),
    ("POL1003", "6407", "Avery Johnson"),
    ("POL1004", "7752", "Sofia Ramirez"),
    ("POL1005", "2219", "Ethan Brooks"),
]


async def init_db() -> None:
    await ensure_pool()
    assert pool is not None
    async with pool.acquire() as conn:
        for statement in SCHEMA_STATEMENTS:
            await conn.execute(statement)
        await _seed_reference_data(conn)


async def close_db() -> None:
    global pool
    if pool is not None:
        await pool.close()
        pool = None


async def database_status() -> str:
    try:
        await fetch_value("SELECT 1")
    except Exception:
        return "unavailable"
    return "ok"


async def fetch_one(query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
    db_pool = await ensure_pool()
    row = await db_pool.fetchrow(query, *params)
    return dict(row) if row else None


async def fetch_all(query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    db_pool = await ensure_pool()
    rows = await db_pool.fetch(query, *params)
    return [dict(row) for row in rows]


async def fetch_value(query: str, params: tuple[Any, ...] = ()) -> Any:
    db_pool = await ensure_pool()
    return await db_pool.fetchval(query, *params)


async def execute(query: str, params: tuple[Any, ...] = ()) -> str:
    db_pool = await ensure_pool()
    return await db_pool.execute(query, *params)


def dump_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


async def ensure_pool() -> asyncpg.Pool:
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(config.DATABASE_URL, min_size=1, max_size=5)
    return pool


async def _seed_reference_data(conn: asyncpg.Connection) -> None:
    await conn.execute("TRUNCATE TABLE claims, policies, verification RESTART IDENTITY CASCADE")
    await conn.executemany(
        """
        INSERT INTO policies(policy_number, holder_name, coverage_type, coverage_limit, deductible, effective_date)
        VALUES ($1, $2, $3, $4, $5, $6)
        ON CONFLICT (policy_number) DO UPDATE SET
            holder_name = EXCLUDED.holder_name,
            coverage_type = EXCLUDED.coverage_type,
            coverage_limit = EXCLUDED.coverage_limit,
            deductible = EXCLUDED.deductible,
            effective_date = EXCLUDED.effective_date
        """,
        SEED_POLICIES,
    )
    await conn.executemany(
        """
        INSERT INTO claims(claim_id, policy_number, status, last_updated, adjuster_name)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (claim_id) DO UPDATE SET
            policy_number = EXCLUDED.policy_number,
            status = EXCLUDED.status,
            last_updated = EXCLUDED.last_updated,
            adjuster_name = EXCLUDED.adjuster_name
        """,
        SEED_CLAIMS,
    )
    await conn.executemany(
        """
        INSERT INTO verification(policy_number, ssn_last4, holder_name)
        VALUES ($1, $2, $3)
        ON CONFLICT (policy_number) DO UPDATE SET
            ssn_last4 = EXCLUDED.ssn_last4,
            holder_name = EXCLUDED.holder_name
        """,
        SEED_VERIFICATION,
    )
