"""Pipeline runner: run Analyzer -> Orchestrator -> Front -> Supervisor and emit events."""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from crewai import Crew

from genie.crew.agents import create_analyzer_agent, create_front_agent, create_supervisor_agent
from genie.crew.tasks import analyzer_task, front_task, supervisor_task
from genie.event_bus import get_event_bus
from genie.orchestrator import RoutingDecision, decide_routing, get_control_layer_for_front
from genie.schemas import (
    AnalyzerOutput,
    ControlLayer,
    EventEnvelope,
    SchemaInfo,
    TokenUsage,
)
from genie.parser import parse_analyzer_output, strip_json_block
from genie.storage.repositories import append_event
from genie.step_id import new_run_id, new_step_id


def run_pipeline(
    run_id: str,
    trace_id: str,
    conversation_id: str,
    user_id: str,
    user_message: str,
    emit_and_store: bool = True,
) -> dict[str, Any]:
    """
    Run the full pipeline synchronously. Emits events to event_bus and stores to DB when emit_and_store=True.
    Returns dict with status, analyzer_output, control_layer, front_reply, supervisor_patch, latency_ms, schema_valid.
    """
    bus = get_event_bus()
    root_step = new_step_id()
    seq = [0]

    def next_seq() -> int:
        s = seq[0]
        seq[0] += 1
        return s

    def emit(
        step_id: str,
        parent_step_id: str,
        component: str,
        event_type: str,
        severity: str = "info",
        latency_ms: int = 0,
        token_usage: Optional[TokenUsage] = None,
        schema: Optional[SchemaInfo] = None,
        payload: Optional[dict] = None,
    ) -> EventEnvelope:
        ev = EventEnvelope(
            run_id=run_id,
            trace_id=trace_id,
            step_id=step_id,
            parent_step_id=parent_step_id,
            sequence=next_seq(),
            ts="",  # set by bus
            component=component,
            event_type=event_type,
            severity=severity,
            latency_ms=latency_ms,
            token_usage=token_usage,
            schema=schema,
            payload=payload or {},
        )
        if emit_and_store:
            from datetime import datetime, timezone
            ev.ts = datetime.now(tz=timezone.utc).isoformat()
            bus.emit(
                run_id=run_id,
                trace_id=trace_id,
                step_id=ev.step_id,
                parent_step_id=ev.parent_step_id,
                sequence=ev.sequence,
                component=ev.component,
                event_type=ev.event_type,
                severity=ev.severity,
                latency_ms=ev.latency_ms,
                token_usage=ev.token_usage,
                schema=ev.schema_info,
                payload=ev.payload,
            )
            append_event(ev)
        return ev

    start_ms = int(time.time() * 1000)
    emit(root_step, "", "orch", "run_started", payload={"user_message": user_message})

    # --- Analyzer ---
    step_analyzer = new_step_id()
    emit(step_analyzer, root_step, "analyzer", "agent_started")
    t0 = time.perf_counter()
    analyzer_agent = create_analyzer_agent()
    crew_analyzer = Crew(agents=[analyzer_agent], tasks=[])
    task_analyzer = analyzer_task(analyzer_agent, user_message, "No prior context for MVP.")
    crew_analyzer = Crew(agents=[analyzer_agent], tasks=[task_analyzer])
    result_analyzer = crew_analyzer.kickoff(inputs={})
    analyzer_raw = result_analyzer.raw if hasattr(result_analyzer, "raw") else str(result_analyzer)
    analyzer_output, schema_info = parse_analyzer_output(analyzer_raw)
    latency_analyzer = int((time.perf_counter() - t0) * 1000)
    emit(
        step_analyzer,
        root_step,
        "analyzer",
        "agent_output",
        latency_ms=latency_analyzer,
        schema=schema_info,
        payload={"raw_preview": analyzer_raw[:500] if analyzer_raw else ""},
    )

    # If schema invalid -> stabilize with safe front reply
    if not schema_info.valid or analyzer_output is None:
        control_for_front = ControlLayer(
            recommended_mode="stabilize",
            block_change_agent=True,
            safety_flag=True,
            safety_reason="Analyzer output invalid or missing.",
        )
        decision = RoutingDecision(mode="stabilize", block_change_agent=True)
    else:
        decision = decide_routing(analyzer_output)
        control_for_front = get_control_layer_for_front(analyzer_output, decision)

    emit(
        new_step_id(),
        root_step,
        "orch",
        "routing_decision",
        payload={"mode": decision.mode, "block_change_agent": decision.block_change_agent},
    )

    # --- Front ---
    step_front = new_step_id()
    emit(step_front, root_step, "front", "agent_started")
    t1 = time.perf_counter()
    front_agent = create_front_agent()
    control_summary = f"mode={control_for_front.recommended_mode}, clarity={control_for_front.clarity_score}, readiness={control_for_front.readiness_score}, resistance_detected={control_for_front.resistance_detected}"
    task_f = front_task(front_agent, control_summary, user_message, decision.style_hint)
    crew_front = Crew(agents=[front_agent], tasks=[task_f])
    result_front = crew_front.kickoff(inputs={})
    front_reply = result_front.raw if hasattr(result_front, "raw") else str(result_front)
    front_reply = front_reply.strip()
    latency_front = int((time.perf_counter() - t1) * 1000)
    emit(
        step_front,
        root_step,
        "front",
        "agent_output",
        latency_ms=latency_front,
        payload={"reply": front_reply},
    )

    # --- Supervisor ---
    step_supervisor = new_step_id()
    emit(step_supervisor, root_step, "supervisor", "agent_started")
    t2 = time.perf_counter()
    run_summary = f"Analyzer valid={schema_info.valid}, Front reply length={len(front_reply)}"
    supervisor_agent = create_supervisor_agent()
    task_sup = supervisor_task(supervisor_agent, run_summary, schema_info.valid, latency_analyzer + latency_front)
    crew_supervisor = Crew(agents=[supervisor_agent], tasks=[task_sup])
    result_supervisor = crew_supervisor.kickoff(inputs={})
    sup_raw = result_supervisor.raw if hasattr(result_supervisor, "raw") else str(result_supervisor)
    sup_raw = sup_raw.strip().lower()
    supervisor_patch = None
    if "no_action" not in sup_raw:
        try:
            supervisor_patch = json.loads(strip_json_block(str(result_supervisor)))
        except Exception:
            supervisor_patch = {"raw": str(result_supervisor)}
    latency_supervisor = int((time.perf_counter() - t2) * 1000)
    emit(
        step_supervisor,
        root_step,
        "supervisor",
        "agent_output",
        latency_ms=latency_supervisor,
        payload={"patch": supervisor_patch},
    )

    total_ms = int(time.time() * 1000) - start_ms
    outputs = {
        "front_reply": front_reply,
        "control_mode": decision.mode,
        "schema_valid": schema_info.valid,
        "supervisor_patch": supervisor_patch,
    }
    if analyzer_output:
        outputs["analyzer_version"] = analyzer_output.version
    emit(new_step_id(), root_step, "orch", "run_completed", latency_ms=total_ms, payload=outputs)
    emit(new_step_id(), root_step, "orch", "stream_end", payload={})

    return {
        "status": "completed",
        "analyzer_output": analyzer_output.model_dump() if analyzer_output else None,
        "control_layer": control_for_front.model_dump(),
        "front_reply": front_reply,
        "supervisor_patch": supervisor_patch,
        "latency_ms": total_ms,
        "schema_valid": schema_info.valid,
        "outputs": outputs,
    }
