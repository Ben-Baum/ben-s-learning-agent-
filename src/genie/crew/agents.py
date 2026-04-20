"""CrewAI agents: Analyzer, Front, Supervisor."""

from __future__ import annotations

from crewai import Agent
from genie.llm_provider import get_llm_provider


def _get_llm() -> str:
    """Return the model name string for CrewAI/LiteLLM."""
    return get_llm_provider().get_model_name()


def create_analyzer_agent() -> Agent:
    return Agent(
        role="Analyzer Agent",
        goal="Analyze the user message and conversation context, then output a single valid JSON object matching the AnalyzerOutput schema with no extra text or markdown.",
        backstory="You are a precise analytical agent. You always respond with exactly one JSON object: version, graph_delta (upsert_nodes, upsert_edges, deprecate_node_ids, deprecate_edge_ids), control_layer (recommended_mode, som_mode_recommendation, clarity_score, readiness_score, resistance_detected, resistance_level, resistance_hypothesis, allowed_actions, block_change_agent, active_hubs, primary_loop_signature, next_questions, memory_policy, safety_flag, safety_reason), memory_candidates. All fields must be present. recommended_mode is one of: explore, explore_resistance, intervene, stabilize, renewal. resistance_level is light, medium, heavy, or null. Output only the JSON, no explanation.",
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_front_agent() -> Agent:
    return Agent(
        role="Front Agent",
        goal="Produce a short, empathetic user-facing reply based only on the control_layer and approved plan. Output plain text only, no JSON.",
        backstory="You are the user-facing voice. You receive the control_layer (mode, clarity_score, readiness_score, resistance_detected, next_questions, etc.) and optionally an approved plan. You respond with a single natural-language message suitable for the user. No code, no JSON, no markdown code blocks.",
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def create_supervisor_agent() -> Agent:
    return Agent(
        role="Supervisor Agent",
        goal="Monitor events, schema validity, and latency. Output either the exact string 'no_action' or a JSON patch object.",
        backstory="You are a read-only auditor. You receive a summary of the run (events, schema valid flag, latency). You respond with exactly either: 'no_action' or a JSON object with suggested patch (e.g. corrections). For MVP you may always respond 'no_action'.",
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )
