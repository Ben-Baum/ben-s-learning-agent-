"""Context manager: conversation state and run context for pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from genie.schemas import AnalyzerOutput, ControlLayer, EventEnvelope, RunInput


@dataclass
class RunContext:
    """Context for a single run."""
    run_id: str
    trace_id: str
    conversation_id: str
    user_id: str
    input: RunInput
    callback_url: Optional[str] = None
    stream_events: bool = True
    # Pipeline outputs
    analyzer_output: Optional[AnalyzerOutput] = None
    control_layer: Optional[ControlLayer] = None
    front_reply: str = ""
    supervisor_patch: Optional[dict[str, Any]] = None
    final_status: str = "running"
    latency_ms: Optional[int] = None
    outputs_json: dict[str, Any] = field(default_factory=dict)
