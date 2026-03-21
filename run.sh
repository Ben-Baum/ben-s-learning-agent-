#!/usr/bin/env bash
# One-command run: install (if needed), run tests, start server.
# Requires Python 3.10+ and .env with GENIE_API_KEY and OPENAI_API_KEY.

set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if ! $PY -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
  echo "Python 3.10+ required. Set PYTHON=python3.10 or install Python 3.10."
  exit 1
fi
if [ ! -d ".venv" ]; then
  $PY -m venv .venv
fi
. .venv/bin/activate
pip install -e ".[dev]" -q
echo "Running tests..."
pytest tests/ -v --tb=short
echo "Starting server..."
exec python -m uvicorn genie.api_server:app --host 0.0.0.0 --port 8000
