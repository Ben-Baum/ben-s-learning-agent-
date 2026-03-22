"""
Genie AI — Optimized Multi-Agent Pipeline

Routes messages through 3 paths based on Smart Router classification:
  light:  Router → Front Agent                          (1 API call)
  medium: Router → NLP → Belief Graph (code) → Front    (2 API calls)
  deep:   Router → NLP → Belief Graph (code) + RAG → Strategy → Front (3 API calls)

Inspired by CrewAI efficiency: not every step needs an LLM call.
"""

from typing import Any, Dict, List, Optional, Tuple

from models import (
    NLPExtractionResult,
    TacticalStrategyResult,
    InvestigationVector,
    StrategyMeta,
)
from prompts import (
    SYSTEM_PROMPT_NLP_ANALYZER,
    SYSTEM_PROMPT_TACTICAL_STRATEGIST,
    FRONT_AGENT_SYSTEM_PROMPT,
)
from llm_client import call_llm_json, call_llm_chat
from graph_utils import apply_delta_to_graph
from knowledge_retriever import retrieve, format_for_prompt
from smart_router import classify_message, RouteType
from belief_graph_rules import compute_belief_graph_delta_rules
from agent_observer import agent_event


# ─── Model Configuration ────────────────────────────────────────────────────
# Switched from Llama 3.3 (Groq) → Gemini 2.5 Flash (Google)
# Reasons: stronger personality, better Hebrew, free tier 1K req/day
# API docs: https://ai.google.dev/gemini-api/docs/openai
DEEP_MODEL = "gemini-2.5-flash"


def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


# ─────────────────────────────────────────────────
#  Step 1: NLP Extraction (API call)
# ─────────────────────────────────────────────────
def run_nlp_extraction(text: str) -> NLPExtractionResult:
    agent_event("nlp_analyzer", "thinking", {"content": text[:60] + "…"}, status="thinking")
    result = call_llm_json(
        model=DEEP_MODEL,
        system_prompt=SYSTEM_PROMPT_NLP_ANALYZER,
        user_content=text,
        response_model=NLPExtractionResult,
    )
    agent_event("nlp_analyzer", "result", {
        "emotions": [{"label": e.label, "intensity": e.intensity, "polarity": e.polarity} for e in result.emotions],
        "cognitive_distortions": [{"type": d.type, "confidence": d.confidence, "explanation": d.explanation_short} for d in result.cognitive_distortions],
        "beliefs": [{"id": b.id, "level": b.level, "valence": b.valence, "statement": b.statement, "strength": b.strength} for b in result.beliefs],
    }, status="idle")
    return result


# ─────────────────────────────────────────────────
#  Step 2: Belief Graph — NOW RULE-BASED (no API!)
# ─────────────────────────────────────────────────
def compute_belief_graph_update(
    *,
    nlp_result: NLPExtractionResult,
    current_graph_json: Dict[str, Any],
) -> Dict[str, Any]:
    """Rule-based belief graph update. No LLM call needed."""
    agent_event("belief_graph", "start", {"task": "Update belief graph", "role": "belief_graph"}, status="active")
    delta = compute_belief_graph_delta_rules(nlp_result)
    updated = apply_delta_to_graph(current_graph_json, delta)
    delta_summary = {"new_nodes": len(delta.new_nodes), "new_edges": len(delta.new_or_updated_edges), "total_nodes": len(updated.get("nodes", {}))}
    agent_event("belief_graph", "done", delta_summary, status="idle")
    return updated


# ─────────────────────────────────────────────────
#  Step 3: Knowledge Retrieval (local DB — no API)
# ─────────────────────────────────────────────────
def retrieve_knowledge(
    user_text: str,
    nlp: NLPExtractionResult,
) -> str:
    """Retrieve relevant expert knowledge from the ingested book library."""
    agent_event("knowledge_retriever", "start", {"task": "RAG retrieval", "role": "knowledge_retriever"}, status="active")
    keywords = []
    for e in nlp.emotions:
        keywords.append(e.label)
    for d in nlp.cognitive_distortions:
        keywords.append(d.type)
    for b in nlp.beliefs:
        keywords.append(b.statement[:30])

    results = retrieve(user_text, nlp_keywords=keywords)
    context = format_for_prompt(results)
    agent_event("knowledge_retriever", "done", {"found": len(results), "keywords": keywords[:5], "context_preview": context[:200] if context else ""}, status="idle")
    return context


# ─────────────────────────────────────────────────
#  Step 4: Tactical Strategy (API call — deep only)
# ─────────────────────────────────────────────────
def compute_tactical_strategy(
    *,
    updated_belief_graph_json: Dict[str, Any],
    recent_nlp_results: Optional[List[NLPExtractionResult]] = None,
    knowledge_context: str = "",
) -> TacticalStrategyResult:
    agent_event("tactician", "thinking", {"content": "Analyzing belief graph + knowledge…"}, status="thinking")
    payload = {
        "updated_belief_graph": updated_belief_graph_json,
        "recent_nlp_results": [r.model_dump() for r in (recent_nlp_results or [])],
    }
    user_content = _json_dumps(payload)
    if knowledge_context:
        user_content += (
            "\n\nEXPERT_KNOWLEDGE (use to inform your investigation vectors — "
            "suggest angles inspired by these sources, but NEVER expose source names):\n"
            + knowledge_context
        )
    result = call_llm_json(
        model=DEEP_MODEL,
        system_prompt=SYSTEM_PROMPT_TACTICAL_STRATEGIST,
        user_content=user_content,
        response_model=TacticalStrategyResult,
    )
    agent_event("tactician", "result", {
        "resistance": result.meta.detected_resistance,
        "vectors": [{"id": v.id, "priority": v.priority, "focus": v.focus_type, "description": v.short_description, "angle": v.suggested_angle_for_front_agent} for v in result.investigation_vectors],
    }, status="idle")
    return result


# ─────────────────────────────────────────────────
#  Step 5: Front Agent (API call — always)
# ─────────────────────────────────────────────────
def front_agent_reply(
    *,
    conversation_history: List[Dict[str, str]],
    strategy: Optional[TacticalStrategyResult] = None,
    nlp_result: Optional[NLPExtractionResult] = None,
) -> str:
    agent_event("front_agent", "thinking", {"content": "Composing response…"}, status="thinking")
    import json
    vectors = []
    if strategy and strategy.investigation_vectors:
        vectors = [v.model_dump() for v in strategy.investigation_vectors]

    user_content = (
        "CONVERSATION_HISTORY:\n"
        f"{json.dumps(conversation_history, ensure_ascii=False, indent=2)}\n\n"
    )

    # Pass NLP insights so the Front Agent isn't blind — even in medium route.
    # This costs nothing extra (no API call) and lets the agent adjust tone,
    # framing, and warmth based on what was actually detected.
    if nlp_result:
        nlp_summary = {
            "emotions": [
                {"label": e.label, "intensity": round(e.intensity, 2), "polarity": e.polarity}
                for e in nlp_result.emotions
                if e.intensity >= 0.3
            ],
            "cognitive_distortions": [
                {"type": d.type, "confidence": round(d.confidence, 2)}
                for d in nlp_result.cognitive_distortions
                if d.confidence >= 0.4
            ],
            "core_beliefs": [
                {"level": b.level, "valence": b.valence, "statement": b.statement}
                for b in nlp_result.beliefs
                if b.strength >= 0.4
            ],
        }
        user_content += (
            "NLP_INSIGHTS (internal — use to calibrate tone and framing only, "
            "never mention or reference directly):\n"
            f"{json.dumps(nlp_summary, ensure_ascii=False, indent=2)}\n\n"
        )

    if vectors:
        user_content += (
            "INVESTIGATION_VECTORS (internal, do NOT show to user):\n"
            f"{json.dumps(vectors, ensure_ascii=False, indent=2)}\n\n"
        )

    user_content += (
        "Now respond ONLY with your next message to the user, "
        "in natural language, following at most ONE of the suggested angles."
    )

    reply = call_llm_chat(
        model=DEEP_MODEL,
        system_prompt=FRONT_AGENT_SYSTEM_PROMPT,
        user_content=user_content,
        temperature=0.7,
    )
    agent_event("front_agent", "result", {"output": reply}, status="idle")
    return reply


# ─────────────────────────────────────────────────
#  EMPTY STRATEGY (for light/medium routes)
# ─────────────────────────────────────────────────
def _empty_strategy() -> TacticalStrategyResult:
    """Return an empty strategy — no investigation vectors."""
    return TacticalStrategyResult(
        meta=StrategyMeta(
            schema_version="1.0",
            detected_resistance=False,
            strongest_signal_belief_ids=[],
            notes_technical="Skipped — light/medium route",
        ),
        investigation_vectors=[],
    )


# ─────────────────────────────────────────────────
#  ROUTE: light — 1 API call
# ─────────────────────────────────────────────────
def _route_light(
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Small talk / short messages → Front Agent only."""
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({"role": "user", "content": user_text})

    reply = front_agent_reply(
        conversation_history=conversation_history,
        strategy=None,
    )
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, {
        **state,
        "conversation_history": conversation_history,
        "last_route": "light",
        "last_api_calls": 1,
    }


# ─────────────────────────────────────────────────
#  ROUTE: medium — 2 API calls
# ─────────────────────────────────────────────────
def _route_medium(
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Regular messages → NLP + Belief Graph (code) + Front Agent."""
    # API call #1: NLP
    nlp = run_nlp_extraction(user_text)

    # Code (no API): Belief Graph update
    belief_graph_json = state.get("belief_graph_json", {})
    updated_graph = compute_belief_graph_update(
        nlp_result=nlp,
        current_graph_json=belief_graph_json,
    )

    # Track NLP history
    recent_nlp = state.get("recent_nlp_results", [])
    recent_nlp = (recent_nlp + [nlp])[-20:]

    # API call #2: Front Agent — now receives NLP insights even in medium route
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({"role": "user", "content": user_text})

    reply = front_agent_reply(
        conversation_history=conversation_history,
        strategy=None,
        nlp_result=nlp,
    )
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, {
        "belief_graph_json": updated_graph,
        "recent_nlp_results": recent_nlp,
        "conversation_history": conversation_history,
        "last_route": "medium",
        "last_api_calls": 2,
    }


# ─────────────────────────────────────────────────
#  ROUTE: deep — 3 API calls
# ─────────────────────────────────────────────────
def _route_deep(
    user_text: str,
    state: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Complex emotional messages → full pipeline with RAG + Strategy."""
    # API call #1: NLP
    nlp = run_nlp_extraction(user_text)

    # Code (no API): Belief Graph update
    belief_graph_json = state.get("belief_graph_json", {})
    updated_graph = compute_belief_graph_update(
        nlp_result=nlp,
        current_graph_json=belief_graph_json,
    )

    # Code (no API): Knowledge Retrieval
    recent_nlp = state.get("recent_nlp_results", [])
    recent_nlp = (recent_nlp + [nlp])[-20:]
    knowledge_context = retrieve_knowledge(user_text, nlp)

    # API call #2: Tactical Strategy (with knowledge)
    strategy = compute_tactical_strategy(
        updated_belief_graph_json=updated_graph,
        recent_nlp_results=recent_nlp[-5:],
        knowledge_context=knowledge_context,
    )

    # API call #3: Front Agent (with strategy vectors)
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({"role": "user", "content": user_text})

    reply = front_agent_reply(
        conversation_history=conversation_history,
        strategy=strategy,
    )
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, {
        "belief_graph_json": updated_graph,
        "recent_nlp_results": recent_nlp,
        "conversation_history": conversation_history,
        "last_route": "deep",
        "last_api_calls": 3,
    }


# ─────────────────────────────────────────────────
#  MAIN ENTRY POINT
# ─────────────────────────────────────────────────
def full_turn(user_text: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Main pipeline entry point. Routes messages intelligently:
      light  → 1 API call   (Front Agent only)
      medium → 2 API calls  (NLP + Front Agent)
      deep   → 3 API calls  (NLP + Strategy + Front Agent)

    Belief Graph is ALWAYS rule-based (0 API calls).
    RAG is ALWAYS local SQLite (0 API calls).
    """
    import time as _time
    turn_id = int(_time.time() * 1000)

    # Signal turn start to dashboard inspector
    agent_event("router", "turn_start", {"id": turn_id, "input": user_text}, status="active")

    route = classify_message(user_text)
    agent_event("router", "start", {"task": f"Route: {route}", "role": "router"}, status="active")
    # Signal route decision — dashboard uses this to highlight path
    agent_event("router", "route_decision", {"route": route}, status="idle")

    if route == "light":
        agent_event("router", "message", {"to": "front_agent", "content": f"light → front_agent"})
        reply, new_state = _route_light(user_text, state)
    elif route == "medium":
        agent_event("router", "message", {"to": "nlp_analyzer", "content": "medium → nlp_analyzer"})
        reply, new_state = _route_medium(user_text, state)
    else:
        agent_event("router", "message", {"to": "nlp_analyzer", "content": "deep → full pipeline"})
        reply, new_state = _route_deep(user_text, state)

    # Signal turn end — dashboard inspector shows the final reply
    agent_event("front_agent", "turn_result", {"text": reply, "route": route}, status="idle")
    return reply, new_state
