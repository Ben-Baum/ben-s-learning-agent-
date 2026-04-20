"""
server.py — Render-compatible server entry point.
Starts the agent_observer HTTP server and keeps the process alive.
Each browser tab gets its own session (conversation state).
"""
import os
import sys
import time
import traceback

# ── Load .env file automatically ──
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            if _line.strip() and not _line.startswith("#"):
                _k, _v = _line.strip().split("=", 1)
                _v = _v.strip("\"'")
                if _k not in os.environ:
                    os.environ[_k] = _v

# agent_observer auto-starts the HTTP server on import
from agent_observer import register_chat_handler, register_ben_agent_handler, register_ben_agent_upload_handler

# ── Persistent Storage Helper ──
import json
USERDATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "userdata")
os.makedirs(USERDATA_DIR, exist_ok=True)

def _get_user_file(user_id: str) -> str:
    # Safely format filename
    safe_id = "".join(c for c in user_id if c.isalnum() or c in "-_")
    return os.path.join(USERDATA_DIR, f"{safe_id}_knowledge.json")

def _load_user(user_id: str) -> dict:
    path = _get_user_file(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading userdata for {user_id}: {e}")
    return {"state": {}, "profile": {}, "uploaded_files": []}

def _save_user(user_id: str, data: dict):
    path = _get_user_file(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving userdata for {user_id}: {e}")

# ── Try to load the pipeline — capture any errors instead of crashing ──
_pipeline_error = None
full_turn = None
ben_agent_full_turn = None

try:
    from pipeline import full_turn, ben_agent_full_turn
    print("✅ Pipeline loaded successfully")
except Exception as e:
    _pipeline_error = f"{type(e).__name__}: {e}"
    print(f"❌ Pipeline failed to load: {_pipeline_error}", file=sys.stderr)
    traceback.print_exc()

# Per-session state: {session_id: state_dict}
_sessions = {}

# Ben's Agent in-memory cache (hydrated from disk)
_ben_agent_sessions = {}


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


def _api_ben_agent_chat(user_text: str, user_id: str) -> dict:
    """Handle Ben's Agent chat — returns {reply, profile}."""
    global _ben_agent_sessions

    if ben_agent_full_turn is None:
        return {"error": f"שגיאה בטעינת הסוכן של בן: {_pipeline_error}"}

    # Load user from dict or disk
    if user_id not in _ben_agent_sessions:
        _ben_agent_sessions[user_id] = _load_user(user_id)
        
    session = _ben_agent_sessions[user_id]
    state = session.get("state", {})
    # Inject uploaded files into state for pipeline to see
    state["uploaded_files"] = session.get("uploaded_files", [])

    try:
        reply, new_state, updated_profile = ben_agent_full_turn(user_text, user_id, state)
    except (OSError, IOError, BrokenPipeError) as io_err:
        # Transient I/O error (e.g. broken stdout pipe from PTY disconnect).
        # Retry once — the pipeline itself is fine, only the logging channel had an issue.
        import sys
        try:
            sys.stdout = open(os.devnull, 'w')
        except Exception:
            pass
        try:
            reply, new_state, updated_profile = ben_agent_full_turn(user_text, user_id, state)
        except Exception as retry_err:
            return {"error": str(retry_err)}
    except Exception as e:
        return {"error": str(e)}

    # Keep conversation history and profile, but preserve uploaded files
    session["state"] = new_state
    session["profile"] = updated_profile

    # Write to memory then disk
    _ben_agent_sessions[user_id] = session
    _save_user(user_id, session)

    return {
        "reply": reply,
        "profile": updated_profile,
        "user_id": user_id,
        "route": new_state.get("last_route", ""),
    }


def _api_ben_agent_upload(file_name: str, mime_type: str, file_data: str, user_id: str) -> dict:
    """Handle Ben's Agent file uploads."""
    global _ben_agent_sessions
    import base64
    import tempfile
    from llm_client import call_llm_chat

    if user_id not in _ben_agent_sessions:
        _ben_agent_sessions[user_id] = _load_user(user_id)
        
    session = _ben_agent_sessions[user_id]
    if "uploaded_files" not in session:
        session["uploaded_files"] = []

    # Extract base64
    b64_str = file_data
    if "base64," in file_data:
        b64_str = file_data.split("base64,")[1]
    
    file_bytes = base64.b64decode(b64_str)
    extracted_text = ""

    print(f"📥 Received file: {file_name} ({mime_type})")

    try:
        if "pdf" in mime_type.lower() or file_name.lower().endswith(".pdf"):
            import pdfplumber
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            
            with pdfplumber.open(tmp_path) as pdf:
                pages = [p.extract_text() for p in pdf.pages if p.extract_text()]
                extracted_text = "\n".join(pages)
                
            os.remove(tmp_path)
            
        elif mime_type.startswith("image/"):
            print(f"👁️ Sending image to Gemini for OCR and analysis...")
            prompt = (
                "You are an expert NLP and psychological observer. Extract all text from this image exactly as written. "
                "If there are diagrams, models, or charts (like CBT cycles, NLP models), describe them in detail. "
                "Output ONLY the extracted text and detailed structural descriptions. No introductory remarks."
            )
            extracted_text = call_llm_chat(
                model="gemini-2.5-flash",
                system_prompt="You are a meticulous transcriber.",
                user_content=prompt,
                image_base64=b64_str,
                mime_type=mime_type,
            )
            extracted_text = f"[IMAGE TRANSLATION AND ANALYSIS: {file_name}]\n{extracted_text}"

        else:
            # Assume text
            extracted_text = file_bytes.decode("utf-8", errors="replace")

    except Exception as e:
        return {"error": f"Failed to parse file: {str(e)}"}

    if not extracted_text.strip():
        return {"error": "No text could be extracted from the file."}

    session["uploaded_files"].append({
        "name": file_name,
        "content": extracted_text
    })

    _ben_agent_sessions[user_id] = session
    _save_user(user_id, session)

    return {"success": True, "file": file_name, "total_files": len(session["uploaded_files"])}


def _ben_agent_get_profile(user_id: str) -> dict:
    """Get a user's learned profile and file list."""
    if user_id not in _ben_agent_sessions:
        _ben_agent_sessions[user_id] = _load_user(user_id)
        
    session = _ben_agent_sessions[user_id]
    profile = session.get("profile", {})
    
    # We send back file names (not content) for the UI
    files = [f["name"] for f in session.get("uploaded_files", [])]
    return {**profile, "uploaded_filenames": files}


def _ben_agent_reset_profile(user_id: str):
    """Reset a user's conversation history but keep files and learned behavior."""
    if user_id not in _ben_agent_sessions:
        _ben_agent_sessions[user_id] = _load_user(user_id)
        
    session = _ben_agent_sessions[user_id]
    # Delete just the conversation history so they can restart talking, 
    # but the AI remembers their style and files
    if "state" in session and "conversation_history" in session["state"]:
        session["state"]["conversation_history"] = []
        
    _save_user(user_id, session)


register_chat_handler(_api_chat)
register_ben_agent_handler(_api_ben_agent_chat)
register_ben_agent_upload_handler(_api_ben_agent_upload)

port = int(os.environ.get("PORT", 8766))
print(f"✅ Genie AI server running on port {port}")
if _pipeline_error:
    print(f"⚠️  Chat will return error: {_pipeline_error}")
else:
    print(f"💬 Chat API ready — dashboard can connect")
    print(f"🧠 Ben's Agent ready — learning mode active")

# Keep the process alive (even if pipeline failed — so we can see the error)
while True:
    time.sleep(1)

