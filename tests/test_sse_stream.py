"""SSE: client receives events in order and ends with stream_end."""

import asyncio
import json
import os
import tempfile
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def temp_db(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("GENIE_DB_PATH", os.path.join(d, "test.db"))
        yield


@pytest.mark.asyncio
async def test_event_bus_emits_in_order_and_ends_with_stream_end():
    from genie.event_bus import EventBus
    from genie.schemas import EventEnvelope

    bus = EventBus()
    run_id = "run-sse-1"
    received = []

    async def consume():
        async for ev in bus.subscribe(run_id):
            received.append(ev)
            if ev.event_type == "stream_end":
                break

    async def produce():
        bus.emit(run_id, "trace-1", "s1", "", 0, "orch", "run_started", payload={"x": 1})
        bus.emit(run_id, "trace-1", "s2", "s1", 1, "analyzer", "agent_output", payload={"x": 2})
        bus.emit(run_id, "trace-1", "s3", "", 2, "orch", "stream_end", payload={})

    await asyncio.gather(consume(), produce())
    assert len(received) == 3
    assert received[0].event_type == "run_started"
    assert received[1].event_type == "agent_output"
    assert received[2].event_type == "stream_end"


@pytest.mark.asyncio
async def test_sse_replay_from_db_ends_with_stream_end():
    from genie.storage.db import get_connection
    from genie.storage.repositories import get_events_by_run_id, append_event
    from genie.schemas import EventEnvelope
    from datetime import datetime, timezone

    run_id = "run-replay-1"
    ts = datetime.now(tz=timezone.utc).isoformat()
    for seq, etype in enumerate(["run_started", "agent_output", "run_completed", "stream_end"]):
        append_event(EventEnvelope(
            run_id=run_id, trace_id=run_id, step_id=f"s{seq}", parent_step_id="",
            sequence=seq, ts=ts, component="orch", event_type=etype, payload={}))
    events = get_events_by_run_id(run_id)
    assert len(events) == 4
    assert events[-1]["event_type"] == "stream_end"
