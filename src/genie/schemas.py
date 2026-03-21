"""Strict Pydantic schemas for Genie pipeline."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# --- Graph delta (analyzer output) ---
class GraphNode(BaseModel):
    id: str
    node_type: str = ""
    text: str = ""
    confidence: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    from_node_id: str = ""
    to_node_id: str = ""
    rel: str = ""
    confidence: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class GraphDelta(BaseModel):
    upsert_nodes: list[GraphNode] = Field(default_factory=list)
    upsert_edges: list[GraphEdge] = Field(default_factory=list)
    deprecate_node_ids: list[str] = Field(default_factory=list)
    deprecate_edge_ids: list[str] = Field(default_factory=list)


# --- Control layer ---
class MemoryPolicy(BaseModel):
    surface_personal_memory: Literal["only_confirmed", "never", "ok"] = "never"
    surface_situational_memory: Literal["ok", "never"] = "never"


class NextQuestion(BaseModel):
    id: str = ""
    text: str = ""
    kind: str = ""


class ControlLayer(BaseModel):
    recommended_mode: Literal["explore", "explore_resistance", "intervene", "stabilize", "renewal"] = "explore"
    som_mode_recommendation: Literal["micro", "hold", "deep"] = "hold"
    clarity_score: float = Field(ge=0.0, le=1.0, default=0.0)
    readiness_score: float = Field(ge=0.0, le=1.0, default=0.0)
    resistance_detected: bool = False
    resistance_level: Optional[Literal["light", "medium", "heavy"]] = None
    resistance_hypothesis: Optional[str] = None
    allowed_actions: list[str] = Field(default_factory=list)
    block_change_agent: bool = False
    active_hubs: list[str] = Field(default_factory=list)
    primary_loop_signature: Optional[str] = None
    next_questions: list[NextQuestion] = Field(default_factory=list)
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    safety_flag: bool = False
    safety_reason: Optional[str] = None


class MemoryCandidate(BaseModel):
    id: str = ""
    title: str = ""
    content: str = ""
    allowed_usage: str = ""
    score: float = 0.0
    meta: dict[str, Any] = Field(default_factory=dict)


class AnalyzerOutput(BaseModel):
    version: int = 1
    graph_delta: GraphDelta = Field(default_factory=GraphDelta)
    control_layer: ControlLayer = Field(default_factory=ControlLayer)
    memory_candidates: list[MemoryCandidate] = Field(default_factory=list)
    debug: Optional[dict[str, Any]] = None


# --- Event envelope (SSE / event bus) ---
class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0
    total: int = 0


class SchemaInfo(BaseModel):
    name: str = ""
    version: int = 0
    valid: bool = True
    errors: list[str] = Field(default_factory=list)


COMPONENTS = Literal["orch", "analyzer", "front", "supervisor", "db", "tools", "renewal"]
EVENT_TYPES = Literal[
    "run_started", "routing_decision", "agent_started", "agent_output",
    "tool_called", "tool_result", "db_write", "run_completed", "run_failed",
    "stream_end", "renewal_started", "renewal_completed"
]
SEVERITIES = Literal["debug", "info", "warn", "error"]


class EventEnvelope(BaseModel):
    run_id: str = ""
    trace_id: str = ""
    step_id: str = ""
    parent_step_id: str = ""
    sequence: int = 0
    ts: str = ""  # ISO string
    component: COMPONENTS = "orch"
    event_type: EVENT_TYPES = "run_started"
    severity: SEVERITIES = "info"
    latency_ms: int = 0
    token_usage: Optional[TokenUsage] = None
    schema_info: Optional[SchemaInfo] = Field(default=None, alias="schema")
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# --- API request/response ---
class RunInput(BaseModel):
    user_message: str = ""
    channel: str = "default"


class RunOptions(BaseModel):
    callback_url: Optional[str] = None
    stream_events: bool = True


class CreateRunRequest(BaseModel):
    conversation_id: str
    user_id: str
    input: RunInput = Field(default_factory=RunInput)
    options: RunOptions = Field(default_factory=RunOptions)
    idempotency_key: Optional[str] = None


class CreateRunResponse(BaseModel):
    run_id: str
    status: str = "started"
    events_url: str = ""


class RunStatusResponse(BaseModel):
    run_id: str
    status: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    latency_ms: Optional[int] = None


class RenewalSnapshotRequest(BaseModel):
    conversation_id: str
    user_id: str
    snapshot_data: dict[str, Any] = Field(default_factory=dict)


class RenewalSnapshotResponse(BaseModel):
    snapshot_id: str
    status: str = "saved"
