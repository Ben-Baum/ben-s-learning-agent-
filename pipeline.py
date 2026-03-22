"""
Genie AI — Optimized Multi-Agent Pipeline

Routes messages through 3 paths based on Smart Router classification:
  light:  Router → Front Agent                           (1 API call)
  medium: Router → NLP → Belief Graph (code) → Front     (2 API calls)
  deep:   Router → NLP → Belief Graph (code) + RAG → Strategy → Front (3 API calls)

Inspired by CrewAI efficiency: not every step needs an LLM call.
"""

import os
import time
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
_KNOWLEDGE_CORE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "knowledge",
    "knowledge_core.md",
)
_KNOWLEDGE_CORE_CACHE: Optional[str] = None


def _json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


def _load_knowledge_core() -> str:
    global _KNOWLEDGE_CORE_CACHE
    if _KNOWLEDGE_CORE_CACHE is not None:
        return _KNOWLEDGE_CORE_CACHE
    try:
        with open(_KNOWLEDGE_CORE_PATH, "r", encoding="utf-8") as f:
            _KNOWLEDGE_CORE_CACHE = f.read().strip()
    except OSError:
        _KNOWLEDGE_CORE_CACHE = ""
    return _KNOWLEDGE_CORE_CACHE


def _conversation_window(conversation_history: List[Dict[str, str]], limit: int = 6) -> List[Dict[str, str]]:
    if limit <= 0:
        return []
    return conversation_history[-limit:]


def _emotion_labels(nlp_result: Optional[NLPExtractionResult]) -> List[str]:
    if not nlp_result:
        return []
    return [e.label for e in nlp_result.emotions if e.intensity >= 0.45]


def _build_front_hint(
    *,
    route: RouteType,
    user_text: str,
    nlp_result: Optional[NLPExtractionResult] = None,
    strategy: Optional[TacticalStrategyResult] = None,
) -> Dict[str, Any]:
    hint = {
        "route": route,
        "tone": "light",
        "depth": "surface" if route == "light" else "present",
        "move": "keep it flowing",
        "response_shape": "short reply",
        "question_mode": "optional",
        "humor": "allowed",
        "warmth": "medium",
        "avoid": [
            "therapy tone",
            "diagnosis",
            "explaining the user to themselves",
            "long analysis",
        ],
    }

    if route == "light":
        hint["move"] = "keep momentum and be easy to reply to"
        hint["response_shape"] = "1 short message"
        return hint

    strongest_emotion = None
    if nlp_result and nlp_result.emotions:
        strongest_emotion = max(nlp_result.emotions, key=lambda e: e.intensity)

    if strongest_emotion:
        if strongest_emotion.label in {"sadness", "shame", "fear", "anxiety", "numbness"}:
            hint["tone"] = "warm"
            hint["warmth"] = "high"
            hint["humor"] = "soft only"
            hint["move"] = "stay close, then add a small angle"
        elif strongest_emotion.label in {"anger", "frustration"}:
            hint["tone"] = "sharp but friendly"
            hint["move"] = "cut through the fog with a precise line"
        elif strongest_emotion.label in {"confusion"}:
            hint["tone"] = "clear"
            hint["move"] = "make it simpler and more grounded"

    if nlp_result and any(d.type in {"catastrophizing", "all_or_nothing", "overgeneralization"} for d in nlp_result.cognitive_distortions):
        hint["move"] = "shrink the drama without sounding dismissive"

    if nlp_result and any(b.valence in {"negative", "mixed"} and b.strength >= 0.65 for b in nlp_result.beliefs):
        hint["depth"] = "deeper"
        hint["response_shape"] = "1-2 short lines and maybe 1 natural question"

    if strategy and strategy.investigation_vectors:
        top_vector = strategy.investigation_vectors[0]
        hint["depth"] = "deeper"
        hint["move"] = top_vector.suggested_angle_for_front_agent
        if top_vector.focus_type == "resistance":
            hint["humor"] = "very gentle"
            hint["question_mode"] = "one soft question at most"
            hint["move"] = "approach gently, no pressure, no fixing"

    text_len = len(user_text.split())
    if text_len > 25 and route != "light":
        hint["response_shape"] = "keep it short, don't match the user's length"

    return hint


def _format_front_hint(hint: Dict[str, Any]) -> str:
    avoid = ", ".join(hint.get("avoid", []))
    return (
        f"route={hint.get('route')}; "
        f"tone={hint.get('tone')}; "
        f"depth={hint.get('depth')}; "
        f"move={hint.get('move')}; "
        f"response_shape={hint.get('response_shape')}; "
        f"question_mode={hint.get('question_mode')}; "
        f"humor={hint.get('humor')}; "
        f"warmth={hint.get('warmth')}; "
        f"avoid={avoid}"
    )


# ─────────────────────────────────────────────────
#  Step 1: NLP Extraction (API call)
# ─────────────────────────────────────────────────
def run_nlp_extraction(text: str) -> NLPExtractionResult:
    agent_event("nlp_analyzer", "thinking", {"content": text[:60] + "…"}, status="thinking")
    system_prompt = SYSTEM_PROMPT_NLP_ANALYZER
    knowledge_core = _load_knowledge_core()
    if knowledge_core:
        system_prompt += (
            "\n\nINTERNAL KNOWLEDGE CORE:\n"
            "Use this only as quiet background guidance for pattern selection.\n"
            "Do not output its language or jargon.\n\n"
            f"{knowledge_core}"
        )
    result = call_llm_json(
        model=DEEP_MODEL,
        system_prompt=system_prompt,
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
    knowledge_core = _load_knowledge_core()
    if knowledge_core:
        user_content += (
            "\n\nINTERNAL_KNOWLEDGE_CORE (background principles only — never quote or expose):\n"
            f"{knowledge_core}"
        )
    if knowledge_context:
        user_content += (
            "\n\nEXPERT_KNOWLEDGE (use to inform your investigation vectors — "
            "suggest angles inspired by these sources, but NEVER expose source names):\n"
            + knowledge_context
        )
    try:
        result = call_llm_json(
            model=DEEP_MODEL,
            system_prompt=SYSTEM_PROMPT_TACTICAL_STRATEGIST,
            user_content=user_content,
            response_model=TacticalStrategyResult,
        )
    except Exception as exc:
        fallback = _empty_strategy()
        fallback.meta.notes_technical = f"Strategy fallback: {type(exc).__name__}: {exc}"
        agent_event(
            "tactician",
            "error",
            {"error": str(exc), "fallback": "empty_strategy"},
            status="error",
        )
        agent_event(
            "tactician",
            "result",
            {"resistance": False, "vectors": [], "fallback_used": True},
            status="idle",
        )
        return fallback

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
    route: RouteType,
    conversation_history: List[Dict[str, str]],
    front_hint: Dict[str, Any],
) -> str:
    agent_event("front_agent", "thinking", {"content": "Composing response…"}, status="thinking")
    import json
    visible_history = _conversation_window(conversation_history)
    user_content = (
        "RECENT_CONVERSATION:\n"
        f"{json.dumps(visible_history, ensure_ascii=False, separators=(',', ':'))}\n\n"
    )
    user_content += (
        "BACKSTAGE_FRONT_HINT:\n"
        f"{_format_front_hint(front_hint)}\n\n"
        f"CURRENT_ROUTE: {route}\n\n"
        "Respond with only the next message to the user. "
        "Use the hint as vibe guidance only. Do not explain it."
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
    front_hint = _build_front_hint(route="light", user_text=user_text)
    agent_event("front_agent", "message", {"front_hint": front_hint}, status="active")

    reply = front_agent_reply(
        route="light",
        conversation_history=conversation_history,
        front_hint=front_hint,
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
    stage_latencies: Dict[str, int] = {}
    t0 = time.perf_counter()
    nlp = run_nlp_extraction(user_text)
    stage_latencies["nlp_ms"] = int((time.perf_counter() - t0) * 1000)

    # Code (no API): Belief Graph update
    t1 = time.perf_counter()
    belief_graph_json = state.get("belief_graph_json", {})
    updated_graph = compute_belief_graph_update(
        nlp_result=nlp,
        current_graph_json=belief_graph_json,
    )
    stage_latencies["graph_ms"] = int((time.perf_counter() - t1) * 1000)

    # Track NLP history
    recent_nlp = state.get("recent_nlp_results", [])
    recent_nlp = (recent_nlp + [nlp])[-20:]

    # API call #2: Front Agent — now receives NLP insights even in medium route
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({"role": "user", "content": user_text})
    front_hint = _build_front_hint(route="medium", user_text=user_text, nlp_result=nlp)
    agent_event(
        "front_agent",
        "message",
        {
            "front_hint": front_hint,
            "detected_emotions": _emotion_labels(nlp),
        },
        status="active",
    )

    t2 = time.perf_counter()
    reply = front_agent_reply(
        route="medium",
        conversation_history=conversation_history,
        front_hint=front_hint,
    )
    stage_latencies["front_ms"] = int((time.perf_counter() - t2) * 1000)
    conversation_history.append({"role": "assistant", "content": reply})
    agent_event("router", "result", {"route": "medium", "latency_ms": stage_latencies}, status="idle")

    return reply, {
        "belief_graph_json": updated_graph,
        "recent_nlp_results": recent_nlp,
        "conversation_history": conversation_history,
        "last_route": "medium",
        "last_api_calls": 2,
        "last_front_hint": front_hint,
        "last_stage_latencies": stage_latencies,
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
    stage_latencies: Dict[str, int] = {}
    t0 = time.perf_counter()
    nlp = run_nlp_extraction(user_text)
    stage_latencies["nlp_ms"] = int((time.perf_counter() - t0) * 1000)

    # Code (no API): Belief Graph update
    t1 = time.perf_counter()
    belief_graph_json = state.get("belief_graph_json", {})
    updated_graph = compute_belief_graph_update(
        nlp_result=nlp,
        current_graph_json=belief_graph_json,
    )
    stage_latencies["graph_ms"] = int((time.perf_counter() - t1) * 1000)

    # Code (no API): Knowledge Retrieval
    t2 = time.perf_counter()
    recent_nlp = state.get("recent_nlp_results", [])
    recent_nlp = (recent_nlp + [nlp])[-20:]
    knowledge_context = retrieve_knowledge(user_text, nlp)
    stage_latencies["rag_ms"] = int((time.perf_counter() - t2) * 1000)

    # API call #2: Tactical Strategy (with knowledge)
    t3 = time.perf_counter()
    strategy = compute_tactical_strategy(
        updated_belief_graph_json=updated_graph,
        recent_nlp_results=recent_nlp[-5:],
        knowledge_context=knowledge_context,
    )
    stage_latencies["strategy_ms"] = int((time.perf_counter() - t3) * 1000)

    # API call #3: Front Agent (with strategy vectors)
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({"role": "user", "content": user_text})
    front_hint = _build_front_hint(
        route="deep",
        user_text=user_text,
        nlp_result=nlp,
        strategy=strategy,
    )
    agent_event(
        "front_agent",
        "message",
        {
            "front_hint": front_hint,
            "strategy_vectors": [v.model_dump() for v in strategy.investigation_vectors[:1]],
            "knowledge_used": bool(knowledge_context),
        },
        status="active",
    )

    t4 = time.perf_counter()
    reply = front_agent_reply(
        route="deep",
        conversation_history=conversation_history,
        front_hint=front_hint,
    )
    stage_latencies["front_ms"] = int((time.perf_counter() - t4) * 1000)
    conversation_history.append({"role": "assistant", "content": reply})
    agent_event("router", "result", {"route": "deep", "latency_ms": stage_latencies}, status="idle")

    return reply, {
        "belief_graph_json": updated_graph,
        "recent_nlp_results": recent_nlp,
        "conversation_history": conversation_history,
        "last_route": "deep",
        "last_api_calls": 3,
        "last_front_hint": front_hint,
        "last_stage_latencies": stage_latencies,
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
