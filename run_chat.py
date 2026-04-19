#!/usr/bin/env python3
"""
Quick REPL to test the full multi-agent pipeline locally.

Usage:
    export GROQ_API_KEY="gsk_..."
    python3 run_chat.py          # normal mode
    python3 run_chat.py --debug  # show pipeline internals
"""

import os
import sys
import json

if not os.environ.get("GOOGLE_API_KEY"):
    print("❌ GOOGLE_API_KEY is not set.")
    print("   Run: export GOOGLE_API_KEY='AIza...'")
    print("   Get a free key at: https://aistudio.google.com")
    sys.exit(1)

DEBUG = "--debug" in sys.argv

from pipeline import (  # noqa: E402
    full_turn,
    ben_agent_full_turn,
    run_nlp_extraction,
    compute_belief_graph_update,
    compute_tactical_strategy,
    retrieve_knowledge,
)
from smart_router import classify_message  # noqa: E402
from graph_utils import apply_delta_to_graph  # noqa: E402
from agent_observer import register_chat_handler  # noqa: E402


_ROUTE_EMOJI = {"light": "🟢", "medium": "🟡", "deep": "🟣"}
_ROUTE_LABEL = {"light": "Light (1 API)", "medium": "Medium (2 APIs)", "deep": "Deep (3 APIs)"}


def _print_debug(state: dict, user_text: str) -> tuple:
    """Run pipeline step-by-step and print each stage."""
    SEP = "─" * 50

    print(f"\n{'═'*50}")
    print("  🔍 PIPELINE DEBUG MODE (Optimized)")
    print(f"{'═'*50}")

    # Step 0: Smart Router
    route = classify_message(
        user_text,
        last_route=state.get("last_route", "light"),
        recent_routes=state.get("recent_routes", []),
    )
    emoji = _ROUTE_EMOJI[route]
    label = _ROUTE_LABEL[route]
    print(f"\n[0] 🧭 Smart Router")
    print(SEP)
    print(f"  מסלול:  {emoji} {label}")
    print(f"  (0 טוקנים — קוד Python)")

    # ── LIGHT route ──
    if route == "light":
        print(f"\n[1/1] Front Agent (Conversational Alchemist)")
        print(SEP)
        print(f"  ⚡ דילוג על NLP, Belief Graph, RAG, Strategy")
        reply, new_state, _ = ben_agent_full_turn(user_text, "debug_user", state)
        print(f"\n  📊 סה\"כ: 1 API call")
        print(f"\n{'═'*50}\n")
        return reply, new_state

    # ── MEDIUM / DEEP: Step 1 — NLP ──
    total_steps = 3 if route == "medium" else 5
    step = 1

    print(f"\n[{step}/{total_steps}] NLP Extraction")
    print(SEP)
    nlp = run_nlp_extraction(user_text)
    emotions = [(e.label, f"{e.intensity:.1f}") for e in nlp.emotions]
    distortions = [d.type for d in nlp.cognitive_distortions]
    beliefs = [(b.level, b.statement[:50]) for b in nlp.beliefs]
    print(f"  רגשות:        {emotions or '—'}")
    print(f"  עיוותים:      {distortions or '—'}")
    print(f"  אמונות:       {beliefs or '—'}")
    print(f"  🔥 API CALL #1")
    step += 1

    # ── MEDIUM / DEEP: Step 2 — Belief Graph (CODE) ──
    print(f"\n[{step}/{total_steps}] Belief Graph Update (rule-based)")
    print(SEP)
    belief_graph_json = state.get("belief_graph_json", {})
    updated_graph = compute_belief_graph_update(
        nlp_result=nlp, current_graph_json=belief_graph_json
    )
    total_nodes = len(updated_graph.get("nodes", {}))
    print(f"  סה\"כ בגרף:   {total_nodes} צמתים")
    print(f"  ⚡ קוד Python — 0 טוקנים")
    step += 1

    recent_nlp = state.get("recent_nlp_results", [])
    recent_nlp = (recent_nlp + [nlp])[-20:]

    strategy = None

    if route == "deep":
        # ── DEEP: Step 3 — RAG ──
        print(f"\n[{step}/{total_steps}] Knowledge Retrieval (RAG)")
        print(SEP)
        knowledge_context = retrieve_knowledge(user_text, nlp)
        if knowledge_context:
            lines = knowledge_context.split("\n")
            for line in lines:
                if line.startswith("["):
                    print(f"  📚 {line}")
                elif line and not line.startswith("KNOWLEDGE"):
                    print(f"     {line[:150]}...")
        else:
            print("  (לא נמצא חומר רלוונטי)")
        print(f"  ⚡ SQLite FTS5 — 0 טוקנים")
        step += 1

        # ── DEEP: Step 4 — Strategy ──
        print(f"\n[{step}/{total_steps}] Tactical Strategy")
        print(SEP)
        strategy = compute_tactical_strategy(
            updated_belief_graph_json=updated_graph,
            recent_nlp_results=recent_nlp[-5:],
            knowledge_context=knowledge_context,
        )
        print(f"  התנגדות:     {'כן' if strategy.meta.detected_resistance else 'לא'}")
        for v in strategy.investigation_vectors:
            print(f"  וקטור [{v.priority}]: {v.short_description}")
            print(f"    → {v.suggested_angle_for_front_agent}")
        print(f"  🔥 API CALL #2")
        step += 1

    # ── MEDIUM / DEEP: Final Step — Front Agent ──
    api_num = 2 if route == "medium" else 3
    print(f"\n[{step}/{total_steps}] Front Agent (Conversational Alchemist)")
    print(SEP)
    reply, new_state, _ = ben_agent_full_turn(user_text, "debug_user", state)
    print(f"  🔥 API CALL #{api_num}")

    new_state = {**new_state, "last_api_calls": api_num}

    print(f"\n  📊 סה\"כ: {api_num} API calls ({_ROUTE_EMOJI[route]} {route})")
    print(f"\n{'═'*50}\n")
    return reply, new_state


def main() -> None:
    mode = "🔬 DEBUG" if DEBUG else "💬 CHAT"
    print("=" * 55)
    print(f"  Genie AI — Conversational Alchemist  [{mode}]")
    print("  Optimized Pipeline: Smart Router + Rule-based BG")
    print("  (type 'quit' or Ctrl+C to exit)")
    if DEBUG:
        print("  --debug: מציג את כל שלבי ה-pipeline")
    print("=" * 55)

    state: dict = {}

    # Register chat handler for the dashboard chat panel
    def _api_chat(user_text: str) -> str:
        nonlocal state
        reply, state = full_turn(user_text, state)
        return reply

    register_chat_handler(_api_chat)
    print("💬  Chat API ready → send messages from the dashboard")

    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nBye 👋")
            break

        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit", "q"}:
            print("Bye 👋")
            break

        print("\nBot: ", end="", flush=True)
        try:
            if DEBUG:
                reply, state = _print_debug(state, user_text)
            else:
                reply, state = full_turn(user_text, state)
            print(reply)

            # Show route info in non-debug mode too
            if not DEBUG and "last_route" in state:
                r = state["last_route"]
                n = state["last_api_calls"]
                print(f"\n  [{_ROUTE_EMOJI[r]} {r} — {n} API calls]")

        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate_limit" in err.lower() or "quota" in err.lower():
                print("\n⚠️  נגמר ה-quota היומי של Groq.")
                print("   זה חינמי — פשוט חכה עד מחר או כנס ל-console.groq.com לראות כמה נשאר.")
                print("   לא תחויב אוטומטית.")
                break
            print(f"[Error] {exc}")


if __name__ == "__main__":
    main()
