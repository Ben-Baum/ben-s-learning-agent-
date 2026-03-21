"""
server.py — Render-compatible server entry point.
Starts the agent_observer HTTP server and keeps the process alive.
"""
import os
import time

# agent_observer auto-starts the HTTP server on import
from agent_observer import register_chat_handler
from pipeline import full_turn

# Register the chat handler so the dashboard can send messages
_state = {}

def _api_chat(user_text: str) -> str:
    global _state
    reply, _state = full_turn(user_text, _state)
    return reply

register_chat_handler(_api_chat)

port = int(os.environ.get("PORT", 8766))
print(f"✅ Genie AI server running on port {port}")
print(f"💬 Chat API ready — dashboard can connect")

# Keep the process alive
while True:
    time.sleep(1)
