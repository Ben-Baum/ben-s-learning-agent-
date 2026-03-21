"""Repository layer for runs, events, idempotency, renewal."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from genie.schemas import EventEnvelope
from genie.storage.db import get_connection


def _now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def ensure_user(user_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)",
            (user_id, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def ensure_conversation(conversation_id: str, user_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR IGNORE INTO conversations (id, user_id, state, created_at) VALUES (?, ?, '{}', ?)",
            (conversation_id, user_id, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def get_run_by_id(run_id: str) -> Optional[dict[str, Any]]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, conversation_id, status, created_at, latency_ms, outputs_json FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "conversation_id": row["conversation_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "latency_ms": row["latency_ms"],
            "outputs_json": json.loads(row["outputs_json"] or "{}"),
        }
    finally:
        conn.close()


def get_run_id_by_idempotency_key(idempotency_key: str) -> Optional[str]:
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT run_id FROM idempotency_keys WHERE idempotency_key = ?",
            (idempotency_key,),
        ).fetchone()
        return row["run_id"] if row else None
    finally:
        conn.close()


def create_run(
    run_id: str,
    conversation_id: str,
    idempotency_key: Optional[str] = None,
) -> None:
    conn = get_connection()
    try:
        now = _now()
        conn.execute(
            "INSERT INTO runs (id, conversation_id, status, created_at, outputs_json) VALUES (?, ?, 'running', ?, '{}')",
            (run_id, conversation_id, now),
        )
        if idempotency_key:
            conn.execute(
                "INSERT OR REPLACE INTO idempotency_keys (idempotency_key, run_id, created_at) VALUES (?, ?, ?)",
                (idempotency_key, run_id, now),
            )
        conn.commit()
    finally:
        conn.close()


def update_run_status(
    run_id: str,
    status: str,
    latency_ms: Optional[int] = None,
    outputs_json: Optional[dict[str, Any]] = None,
) -> None:
    conn = get_connection()
    try:
        if outputs_json is not None:
            conn.execute(
                "UPDATE runs SET status = ?, latency_ms = ?, outputs_json = ? WHERE id = ?",
                (status, latency_ms, json.dumps(outputs_json), run_id),
            )
        else:
            conn.execute(
                "UPDATE runs SET status = ?, latency_ms = ? WHERE id = ?",
                (status, latency_ms, run_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_events_by_run_id(run_id: str) -> list[dict]:
    """Return events for a run ordered by sequence (for SSE replay)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """SELECT run_id, step_id, parent_step_id, sequence, component, event_type, ts, payload_json
               FROM events WHERE run_id = ? ORDER BY sequence""",
            (run_id,),
        ).fetchall()
        out = []
        for r in rows:
            out.append({
                "run_id": r["run_id"],
                "trace_id": r["run_id"],
                "step_id": r["step_id"],
                "parent_step_id": r["parent_step_id"],
                "sequence": r["sequence"],
                "ts": r["ts"],
                "component": r["component"],
                "event_type": r["event_type"],
                "severity": "info",
                "latency_ms": 0,
                "payload": json.loads(r["payload_json"] or "{}"),
            })
        return out
    finally:
        conn.close()


def append_event(ev: EventEnvelope) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO events (run_id, sequence, step_id, parent_step_id, component, event_type, ts, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ev.run_id,
                ev.sequence,
                ev.step_id,
                ev.parent_step_id,
                ev.component,
                ev.event_type,
                ev.ts,
                json.dumps(ev.payload),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def save_renewal_snapshot(
    snapshot_id: str,
    conversation_id: str,
    user_id: str,
    snapshot_data: dict[str, Any],
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO renewal_snapshots (id, conversation_id, user_id, snapshot_data, created_at) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, conversation_id, user_id, json.dumps(snapshot_data), _now()),
        )
        conn.commit()
    finally:
        conn.close()
