"""SQLite connection and initialization."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

# Default DB path: project root / data / genie.db
def _default_db_path() -> str:
    base = os.environ.get("GENIE_DB_PATH")
    if base:
        return base
    root = Path(__file__).resolve().parent.parent.parent.parent  # genie/storage -> genie -> src -> project
    return str(root / "data" / "genie.db")


def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite DB. Creates DB and runs migrations if needed."""
    path = db_path or _default_db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _ensure_migrations(conn)
    return conn


def _ensure_migrations(conn: sqlite3.Connection) -> None:
    migrations_file = Path(__file__).parent / "migrations.sql"
    if migrations_file.exists():
        sql = migrations_file.read_text()
        conn.executescript(sql)
        conn.commit()
