"""Event bus: emits graph-ready events with step_id, parent_step_id, SSE streaming."""

from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Optional

from genie.schemas import EventEnvelope, SchemaInfo, TokenUsage


class EventBus:
    """In-memory event bus; subscribers get EventEnvelope via async iterator. Emit is sync so pipeline can call it."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[EventEnvelope]]] = {}
        self._lock = threading.Lock()

    async def subscribe(self, run_id: str) -> AsyncIterator[EventEnvelope]:
        """Subscribe to events for a run. Yields until stream_end."""
        queue: asyncio.Queue[EventEnvelope] = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(queue)
        try:
            while True:
                ev = await queue.get()
                yield ev
                if ev.event_type == "stream_end":
                    break
        finally:
            with self._lock:
                if run_id in self._subscribers:
                    self._subscribers[run_id].remove(queue)
                    if not self._subscribers[run_id]:
                        del self._subscribers[run_id]

    def emit(
        self,
        run_id: str,
        trace_id: str,
        step_id: str,
        parent_step_id: str,
        sequence: int,
        component: str,
        event_type: str,
        severity: str = "info",
        latency_ms: int = 0,
        token_usage: Optional[TokenUsage] = None,
        schema: Optional[SchemaInfo] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> EventEnvelope:
        """Emit one event and fan-out to subscribers (sync)."""
        ts = datetime.now(tz=timezone.utc).isoformat()
        ev = EventEnvelope(
            run_id=run_id,
            trace_id=trace_id,
            step_id=step_id,
            parent_step_id=parent_step_id,
            sequence=sequence,
            ts=ts,
            component=component,
            event_type=event_type,
            severity=severity,
            latency_ms=latency_ms,
            token_usage=token_usage,
            schema=schema,
            payload=payload or {},
        )
        with self._lock:
            for q in self._subscribers.get(run_id, []):
                q.put_nowait(ev)
        return ev


# Global singleton for the server
_event_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus
