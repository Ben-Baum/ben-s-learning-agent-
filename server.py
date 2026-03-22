"""
server.py — Render-compatible server entry point.
Starts the agent_observer HTTP server and keeps the process alive.
Each browser tab gets its own session (conversation state).
"""
import os
import sys
import time
import traceback

# agent_observer auto-starts the HTTP server on import
from agent_observer import register_chat_handler

# ── Try to load the pipeline — capture any errors instead of crashing ──
_pipeline_error = None
full_turn = None

try:
    from pipeline import full_turn
    print("✅ Pipeline loaded successfully")
except Exception as e:
    _pipeline_error = f"{type(e).__name__}: {e}"
    print(f"❌ Pipeline failed to load: {_pipeline_error}", file=sys.stderr)
    traceback.print_exc()

# Per-session state: {session_id: state_dict}
_sessions = {}

def _api_chat(user_text: str, session_id: str = None) -> str:
    global _sessions

    # If pipeline didn't load, return the actual error
    if full_turn is None:
        return f"שגיאה בטעינת הצינור: {_pipeline_error}"

    state = _sessions.get(session_id, {}) if session_id else {}
    reply, new_state = full_turn(user_text, state)
    if session_id:
        _sessions[session_id] = new_state
    return reply

register_chat_handler(_api_chat)

port = int(os.environ.get("PORT", 8766))
print(f"✅ Genie AI server running on port {port}")
if _pipeline_error:
    print(f"⚠️  Chat will return error: {_pipeline_error}")
else:
    print(f"💬 Chat API ready — dashboard can connect")

# Keep the process alive (even if pipeline failed — so we can see the error)
while True:
    time.sleep(1)
