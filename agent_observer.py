"""
agent_observer.py — Agent Control Center backend.

Provides:
  1. HTTP API server (port 8766) — REST endpoints + SSE real-time event stream
  2. SSE (Server-Sent Events) at /api/events — streams real-time agent events to the dashboard

Usage:
    from agent_observer import agent_event
    agent_event("nlp_analyzer", "thinking", {"content": "..."}, status="thinking")
"""

import json
import os
import re
import threading
import time
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer as _HTTPServer
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import urlparse

# ─── Config ───────────────────────────────────────────────────────────────────
HTTP_PORT = int(os.environ.get("PORT", 8766))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_http_thread: Optional[threading.Thread] = None

# ─── Chat bridge ──────────────────────────────────────────────────────────────
_chat_handler = None   # callable(user_text: str) -> str  (set by run_chat.py)

def register_chat_handler(fn):
    """Register a function that processes user messages and returns a reply."""
    global _chat_handler
    _chat_handler = fn


# ─── SSE Event Bus (replaces WebSocket) ──────────────────────────────────────
#
# How it works:
#   1. _emit(data) pushes events into _sse_queues (one deque per connected SSE client)
#   2. GET /api/events is a long-running HTTP response that streams these events
#   3. The dashboard connects with new EventSource('/api/events')
#   4. No asyncio, no websockets library, no extra ports — just HTTP
#

_sse_lock = threading.Lock()
_sse_queues: list = []   # list of deque objects, one per connected SSE client

# Also keep last 50 events for late-connecting dashboards
_event_history = deque(maxlen=50)


def _emit(data: dict):
    """Thread-safe push to all connected SSE clients."""
    msg = json.dumps(data, ensure_ascii=False)
    with _sse_lock:
        _event_history.append(msg)
        dead = []
        for i, q in enumerate(_sse_queues):
            try:
                q.append(msg)
            except Exception:
                dead.append(i)
        for i in reversed(dead):
            _sse_queues.pop(i)


def _register_sse_client():
    """Register a new SSE client and return its event queue."""
    q = deque(maxlen=500)
    # Send recent history so the dashboard catches up
    with _sse_lock:
        for msg in _event_history:
            q.append(msg)
        _sse_queues.append(q)
    return q


def _unregister_sse_client(q):
    """Remove an SSE client's queue."""
    with _sse_lock:
        try:
            _sse_queues.remove(q)
        except ValueError:
            pass


# ─── Multi-threaded HTTP Server ───────────────────────────────────────────────
class HTTPServer(ThreadingMixIn, _HTTPServer):
    """Multi-threaded HTTP server — each request gets its own thread."""
    daemon_threads = True


class APIHandler(BaseHTTPRequestHandler):
    """Serves dashboard + REST API + SSE event stream."""

    def log_message(self, format, *args):
        pass  # suppress logs

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _html_response(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self._cors()
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/dashboard":
            self._serve_dashboard()
        elif path == "/api/events":
            self._serve_sse()
        elif path == "/api/prompts":
            self._get_prompts()
        elif path == "/api/router-config":
            self._get_router_config()
        elif path == "/api/pipeline-info":
            self._get_pipeline_info()
        elif path == "/api/sse-status":
            self._get_sse_status()
        elif path == "/api/ping":
            self._ping()
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self._json_response({"error": "Invalid JSON"}, 400)
            return

        if path == "/api/prompts":
            self._save_prompts(data)
        elif path == "/api/router-config":
            self._save_router_config(data)
        elif path == "/api/chat":
            self._handle_chat(data)
        else:
            self.send_error(404)

    # ── SSE Event Stream ──
    def _serve_sse(self):
        """Long-running HTTP response that streams SSE events to the dashboard."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()

        q = _register_sse_client()
        try:
            # Send an initial "connected" event
            self.wfile.write(b"data: {\"type\":\"connected\",\"agent\":\"system\",\"status\":\"idle\",\"data\":{}}\n\n")
            self.wfile.flush()

            while True:
                if q:
                    msg = q.popleft()
                    self.wfile.write(f"data: {msg}\n\n".encode("utf-8"))
                    self.wfile.flush()
                else:
                    # Send heartbeat every 2 seconds to keep connection alive
                    time.sleep(0.05)
                    # Check for events frequently but send heartbeat less often
                    if not q:
                        try:
                            self.wfile.write(b": heartbeat\n\n")
                            self.wfile.flush()
                        except Exception:
                            break
                        time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # client disconnected
        finally:
            _unregister_sse_client(q)

    # ── Dashboard ──
    def _serve_dashboard(self):
        dashboard_path = os.path.join(BASE_DIR, "agent_dashboard.html")
        if os.path.exists(dashboard_path):
            with open(dashboard_path, "r", encoding="utf-8") as f:
                self._html_response(f.read())
        else:
            self._html_response("<h1>agent_dashboard.html not found</h1>")

    # ── Chat API ──
    def _handle_chat(self, data):
        user_text = data.get("message", "").strip()
        if not user_text:
            self._json_response({"error": "empty message"}, 400)
            return
        if _chat_handler is None:
            self._json_response({"error": "pipeline not running"}, 503)
            return
        try:
            reply = _chat_handler(user_text)
            self._json_response({"reply": reply})
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    # ── SSE Status & Ping ──
    def _get_sse_status(self):
        with _sse_lock:
            count = len(_sse_queues)
        self._json_response({
            "connected_clients": count,
            "history_size": len(_event_history),
        })

    def _ping(self):
        """Fire a fake full-pipeline event sequence to test SSE connectivity."""
        def fire():
            steps = [
                (0.0,  "router",             "turn_start",    {"id": 9999, "input": "⚡ PING TEST"}, "active"),
                (0.25, "router",             "start",         {"task": "Route: deep", "role": "router"}, "active"),
                (0.5,  "router",             "route_decision",{"route": "deep"}, "idle"),
                (0.8,  "nlp_analyzer",       "thinking",      {"content": "Analyzing emotions..."}, "thinking"),
                (1.3,  "nlp_analyzer",       "result",        {"emotions": [{"label": "sadness", "intensity": 0.8, "polarity": "negative"}], "cognitive_distortions": [{"type": "catastrophizing", "confidence": 0.7, "explanation": "Expecting worst"}], "beliefs": [{"id": "b1", "level": "intermediate", "valence": "negative", "statement": "אני לא מספיק טוב", "strength": 0.8}]}, "idle"),
                (1.5,  "belief_graph",       "start",         {"task": "Update belief graph"}, "active"),
                (1.8,  "belief_graph",       "done",          {"new_nodes": 2, "new_edges": 1, "total_nodes": 5}, "idle"),
                (2.0,  "knowledge_retriever","start",         {"task": "RAG retrieval"}, "active"),
                (2.3,  "knowledge_retriever","done",          {"found": 3, "keywords": ["sadness", "catastrophizing", "self-worth"], "context_preview": "CBT techniques for addressing negative core beliefs..."}, "idle"),
                (2.5,  "tactician",          "thinking",      {"content": "Building strategy..."}, "thinking"),
                (3.0,  "tactician",          "result",        {"resistance": False, "vectors": [{"id": "v1", "priority": "high", "focus": "cognitive_restructuring", "description": "Challenge catastrophizing pattern", "angle": "gentle Socratic questioning"}]}, "idle"),
                (3.2,  "front_agent",        "thinking",      {"content": "Composing response..."}, "thinking"),
                (3.7,  "front_agent",        "turn_result",   {"text": "✅ SSE is working! Events are flowing correctly through the entire pipeline.", "route": "deep"}, "idle"),
            ]
            start = time.time()
            for delay, agent, etype, data, status in steps:
                elapsed = time.time() - start
                remaining = delay - elapsed
                if remaining > 0:
                    time.sleep(remaining)
                agent_event(agent, etype, data, status=status)

        t = threading.Thread(target=fire, daemon=True)
        t.start()
        self._json_response({"status": "ping sequence started", "steps": 13})

    # ── Prompts API ──
    def _get_prompts(self):
        prompts_path = os.path.join(BASE_DIR, "prompts.py")
        prompts = self._parse_prompts(prompts_path)
        self._json_response(prompts)

    def _parse_prompts(self, path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        prompts = {}
        pattern = r'(\w+)\s*=\s*"""(.*?)"""'
        for match in re.finditer(pattern, content, re.DOTALL):
            name = match.group(1)
            text = match.group(2).strip()
            prompts[name] = text
        return prompts

    def _save_prompts(self, data):
        prompts_path = os.path.join(BASE_DIR, "prompts.py")
        with open(prompts_path, "r", encoding="utf-8") as f:
            content = f.read()
        for name, new_text in data.items():
            pattern = rf'({re.escape(name)}\s*=\s*""").*?(""")'
            replacement = rf'\g<1>\n{new_text}\n\g<2>'
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        with open(prompts_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._json_response({"status": "saved", "updated": list(data.keys())})

    # ── Router Config API ──
    def _get_router_config(self):
        router_path = os.path.join(BASE_DIR, "smart_router.py")
        config = self._parse_router(router_path)
        self._json_response(config)

    def _parse_router(self, path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        config = {
            "light_patterns": [],
            "deep_signals": [],
            "emotional_keywords_he": [],
            "emotional_keywords_en": [],
        }
        light_match = re.search(r'_LIGHT_PATTERNS\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if light_match:
            config["light_patterns"] = re.findall(r'r"(.*?)"', light_match.group(1))
        deep_match = re.search(r'_DEEP_SIGNALS\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if deep_match:
            config["deep_signals"] = re.findall(r'r"(.*?)"', deep_match.group(1))
        he_match = re.search(r'emotional_keywords_he\s*=\s*\{(.*?)\}', content, re.DOTALL)
        if he_match:
            config["emotional_keywords_he"] = re.findall(r'"(.*?)"', he_match.group(1))
        en_match = re.search(r'emotional_keywords_en\s*=\s*\{(.*?)\}', content, re.DOTALL)
        if en_match:
            config["emotional_keywords_en"] = re.findall(r'"(.*?)"', en_match.group(1))
        return config

    def _save_router_config(self, data):
        router_path = os.path.join(BASE_DIR, "smart_router.py")
        with open(router_path, "r", encoding="utf-8") as f:
            content = f.read()

        if "light_patterns" in data:
            patterns_str = ",\n    ".join([f'r"{p}"' for p in data["light_patterns"]])
            content = re.sub(
                r'_LIGHT_PATTERNS\s*=\s*\[.*?\]',
                f'_LIGHT_PATTERNS = [\n    {patterns_str},\n]',
                content, flags=re.DOTALL
            )
        if "deep_signals" in data:
            signals_str = ",\n    ".join([f'r"{p}"' for p in data["deep_signals"]])
            content = re.sub(
                r'_DEEP_SIGNALS\s*=\s*\[.*?\]',
                f'_DEEP_SIGNALS = [\n    {signals_str},\n]',
                content, flags=re.DOTALL
            )
        if "emotional_keywords_he" in data:
            kw_str = ", ".join([f'"{k}"' for k in data["emotional_keywords_he"]])
            content = re.sub(
                r'emotional_keywords_he\s*=\s*\{.*?\}',
                f'emotional_keywords_he = {{\n        {kw_str},\n    }}',
                content, flags=re.DOTALL
            )
        if "emotional_keywords_en" in data:
            kw_str = ", ".join([f'"{k}"' for k in data["emotional_keywords_en"]])
            content = re.sub(
                r'emotional_keywords_en\s*=\s*\{.*?\}',
                f'emotional_keywords_en = {{\n        {kw_str},\n    }}',
                content, flags=re.DOTALL
            )

        with open(router_path, "w", encoding="utf-8") as f:
            f.write(content)
        self._json_response({"status": "saved"})

    # ── Pipeline Info ──
    def _get_pipeline_info(self):
        self._json_response({
            "agents": [
                {"id": "router", "role": "Smart Router", "type": "code", "description": "מסווג הודעות ל-light/medium/deep", "prompt_key": None},
                {"id": "nlp_analyzer", "role": "NLP Analyzer", "type": "llm", "description": "חילוץ רגשות, עיוותים ואמונות", "prompt_key": "SYSTEM_PROMPT_NLP_ANALYZER"},
                {"id": "belief_graph", "role": "Belief Graph", "type": "code", "description": "עדכון גרף אמונות (rule-based)", "prompt_key": "SYSTEM_PROMPT_BELIEF_GRAPH_MAPPER"},
                {"id": "knowledge_retriever", "role": "Knowledge Retriever", "type": "code", "description": "חיפוש RAG ב-SQLite", "prompt_key": None},
                {"id": "tactician", "role": "Tactical Strategist", "type": "llm", "description": "יצירת כיוון פנימי עדין לשיחה", "prompt_key": "SYSTEM_PROMPT_TACTICAL_STRATEGIST"},
                {"id": "front_agent", "role": "Front Agent", "type": "llm", "description": "תגובה סופית למשתמש בעברית לפי front hint קצר", "prompt_key": "FRONT_AGENT_SYSTEM_PROMPT"},
            ],
            "routes": {
                "light": {"agents": ["router", "front_agent"], "api_calls": 1},
                "medium": {"agents": ["router", "nlp_analyzer", "belief_graph", "front_agent"], "api_calls": 2},
                "deep": {"agents": ["router", "nlp_analyzer", "belief_graph", "knowledge_retriever", "tactician", "front_agent"], "api_calls": 3},
            },
            "model": "gemini-2.5-flash",
            "provider": "Google Gemini via OpenAI-compatible endpoint",
        })


def _run_http_server():
    host = os.environ.get("HOST", "0.0.0.0")
    server = HTTPServer((host, HTTP_PORT), APIHandler)
    server.serve_forever()


# ─── Public API ───────────────────────────────────────────────────────────────

def start_server():
    """Call once at the start of your script."""
    global _http_thread

    if not _http_thread or not _http_thread.is_alive():
        _http_thread = threading.Thread(target=_run_http_server, daemon=True)
        _http_thread.start()

    print(f"🔭  Agent Control Center → http://localhost:{HTTP_PORT}")
    print(f"📡  SSE event stream     → http://localhost:{HTTP_PORT}/api/events")


def agent_event(agent_id: str, event_type: str, data: dict = None, status: str = "active"):
    """
    Emit any event from any agent.
    event_type: start | thinking | tool_use | message | result | error | done
    """
    _emit({
        "agent": agent_id,
        "type": event_type,
        "data": data or {},
        "status": status,
        "ts": datetime.utcnow().isoformat(),
    })


@contextmanager
def observe(agent_id: str, task: str = "", role: str = ""):
    """Context manager that auto-emits start/done/error events."""
    agent_event(agent_id, "start", {"task": task, "role": role}, status="active")
    try:
        yield
        agent_event(agent_id, "done", {"task": task}, status="idle")
    except Exception as e:
        agent_event(agent_id, "error", {"error": str(e), "task": task}, status="error")
        raise


# ─── Auto-start ───────────────────────────────────────────────────────────────
start_server()
