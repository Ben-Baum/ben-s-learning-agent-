"""CrewAI tasks for Analyzer, Front, Supervisor."""

from __future__ import annotations

from crewai import Agent, Task


def analyzer_task(agent: Agent, user_message: str, context_summary: str) -> Task:
    return Task(
        description=f"""Analyze this user message and context. Output a single JSON object (AnalyzerOutput).
User message: {user_message}
Context: {context_summary}

Required JSON shape (all fields required): version (1), graph_delta (upsert_nodes, upsert_edges, deprecate_node_ids, deprecate_edge_ids), control_layer (recommended_mode: explore|explore_resistance|intervene|stabilize|renewal, som_mode_recommendation: micro|hold|deep, clarity_score 0-1, readiness_score 0-1, resistance_detected bool, resistance_level: light|medium|heavy|null, resistance_hypothesis string|null, allowed_actions list, block_change_agent bool, active_hubs list, primary_loop_signature string|null, next_questions list of {{id, text, kind}}, memory_policy (surface_personal_memory: only_confirmed|never|ok, surface_situational_memory: ok|never), safety_flag bool, safety_reason string|null), memory_candidates list.
Output ONLY the JSON, no other text.""",
        expected_output="A single valid JSON object with version, graph_delta, control_layer, memory_candidates.",
        agent=agent,
    )


def front_task(
    agent: Agent,
    control_layer_summary: str,
    user_message: str,
    style_hint: str = "engaging",
) -> Task:
    return Task(
        description=f"""Using ONLY the control layer and the user message, write a short user-facing reply.
Control layer: {control_layer_summary}
User message: {user_message}
Style: {style_hint}. Be warm and concise.
Output only the reply text, no JSON or markdown.""",
        expected_output="Plain text user-facing message.",
        agent=agent,
    )


def supervisor_task(
    agent: Agent,
    run_summary: str,
    schema_valid: bool,
    latency_ms: int,
) -> Task:
    return Task(
        description=f"""Audit this run. Run summary: {run_summary}. Schema valid: {schema_valid}. Latency ms: {latency_ms}.
Respond with exactly either: no_action (if all ok) or a JSON patch object if you suggest corrections.""",
        expected_output="Either the string 'no_action' or a JSON object.",
        agent=agent,
    )
