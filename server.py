"""
server.py — Render-compatible server entry point.
Starts the agent_observer HTTP server and keeps the process alive.
Each browser tab gets its own session (conversation state).
"""
import os
import time

# agent_observer auto-starts the HTTP server on import
from agent_observer import register_chat_handler
from pipeline import full_turn

# Per-session state: {session_id: state_dict}
_sessions = {}

def _api_chat(user_text: str, session_id: str = None) -> str:
    global _sessions
    state = _sessions.get(session_id, {}) if session_id else {}
    reply, new_state = full_turn(user_text, state)
    if session_id:
        _sessions[session_id] = new_state
    return reply

register_chat_handler(_api_chat)

port = int(os.environ.get("PORT", 8766))
print(f"✅ Genie AI server running on port {port}")
print(f"💬 Chat API ready — dashboard can connect")

# Keep the process alive
while True:
    time.sleep(1)
