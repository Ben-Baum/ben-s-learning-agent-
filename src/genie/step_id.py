"""Step ID generation for trace hierarchy."""

import uuid
from typing import Optional


def new_step_id(parent_step_id: Optional[str] = None) -> str:
    """Generate a unique step_id. Optional parent for hierarchy."""
    return str(uuid.uuid4())


def new_run_id() -> str:
    """Generate a unique run_id."""
    return str(uuid.uuid4())
