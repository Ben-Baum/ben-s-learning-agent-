# Genie Multi-Agent System

Backend service that runs a multi-agent pipeline: **Analyzer** → **Orchestrator** (deterministic) → **Front** → **Supervisor**, with an event bus and SSE streaming for live execution.

## Setup

### 1. Create virtualenv and install

```bash
cd genie
python3.10 -m venv .venv  # or python3.11; requires 3.10+
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Environment variables

Copy `.env.example` to `.env` and set:

| Variable | Description |
|----------|-------------|
| `LLM_PROVIDER` | `openai` or `gemini` (default: `openai`) |
| `OPENAI_API_KEY` | Required when `LLM_PROVIDER=openai` |
| `GEMINI_API_KEY` | For Gemini (stub; use OpenAI for MVP) |
| `GENIE_API_KEY` | Required for Bridge API auth (`Authorization: Bearer <key>`) |
| `GENIE_DB_PATH` | Optional; default: `./data/genie.db` |
| `GENIE_HOST` / `GENIE_PORT` | Server bind (default `0.0.0.0:8000`) |
| `GENIE_BASE_URL` | Optional base URL for `events_url` in run response |

### 3. Run the server

```bash
python -m uvicorn genie.api_server:app --host 0.0.0.0 --port 8000
# or
genie-serve
```

**One command** (install, test, then start server; requires Python 3.10+):

```bash
cd genie
cp .env.example .env   # then set GENIE_API_KEY and OPENAI_API_KEY
./run.sh
```

Or step by step: create venv with Python 3.10+, `pip install -e ".[dev]"`, then `pytest tests/ -v` and `python -m uvicorn genie.api_server:app --host 0.0.0.0 --port 8000`.

## Example: Create run and connect to SSE

```bash
# Create a run (requires GENIE_API_KEY in .env)
curl -X POST http://localhost:8000/v1/runs \
  -H "Authorization: Bearer YOUR_GENIE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"conv-1","user_id":"user-1","input":{"user_message":"Hello, I want to understand my goals."}}'

# Response: {"run_id":"<uuid>","status":"started","events_url":"/v1/runs/<run_id>/events"}

# Stream events (SSE)
curl -N -H "Authorization: Bearer YOUR_GENIE_API_KEY" \
  "http://localhost:8000/v1/runs/<run_id>/events"

# Get run status and outputs
curl -H "Authorization: Bearer YOUR_GENIE_API_KEY" \
  "http://localhost:8000/v1/runs/<run_id>"
```

## Switching LLM provider

- **OpenAI (default):** Set `LLM_PROVIDER=openai` and `OPENAI_API_KEY`. Uses `gpt-4o-mini` unless `OPENAI_MODEL` is set.
- **Gemini (stub):** Set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY`. The codebase includes a `GeminiProvider` stub in `src/genie/llm_provider.py`; full integration requires wiring the Gemini SDK to CrewAI (e.g. custom LLM class or adapter). For MVP, use OpenAI.

## Architecture (MVP)

- **Analyzer Agent (LLM):** Outputs strict JSON: `AnalyzerOutput` (graph_delta, control_layer, memory_candidates).
- **Front Agent (LLM):** User-facing reply from control_layer only.
- **Supervisor Agent (LLM):** Audits run; outputs `no_action` or patch JSON (read-only).
- **Orchestrator (non-LLM):** Resistance-first rules: `safety_flag` → stabilize; `resistance_detected` → explore_resistance; `readiness_score >= 0.65` → intervene; else explore.
- **Event Bus:** In-memory; events stored in SQLite; SSE streams live or replays from DB.
- **Chat Renewal:** `POST /v1/renewal/snapshot` saves a snapshot (stub).

## Known limitations and next steps

- **Change Agent:** Not implemented; Front produces reply only; `block_change_agent` is respected by orchestrator.
- **Database:** SQLite for MVP; replace with Postgres for production (connection pool, migrations).
- **RAG / memory:** Memory candidates and belief graph tables exist; no RAG or graph updates in pipeline yet.
- **Gemini:** Adapter is a stub; use OpenAI for a working MVP.
- **Cancellation:** `POST /v1/runs/{run_id}/cancel` marks run as cancelled but does not stop an in-flight pipeline (no task handle).

## Running tests

```bash
cd genie
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

Tests include: schema validation (and invalid JSON fallback), orchestrator routing rules, idempotency (same key → same run_id), and SSE order + stream_end.
