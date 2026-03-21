"""Parse analyzer LLM output to AnalyzerOutput. No CrewAI dependency."""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

from genie.schemas import AnalyzerOutput, SchemaInfo


def strip_json_block(raw: str) -> str:
    """Extract JSON from LLM output (handle markdown code blocks)."""
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        return m.group(1).strip()
    return raw


def parse_analyzer_output(raw: str) -> Tuple[Optional[AnalyzerOutput], SchemaInfo]:
    """Parse analyzer string to AnalyzerOutput. Return (output or None, schema_info)."""
    try:
        text = strip_json_block(raw)
        data = json.loads(text)
        out = AnalyzerOutput.model_validate(data)
        return out, SchemaInfo(name="AnalyzerOutput", version=1, valid=True, errors=[])
    except Exception as e:
        return None, SchemaInfo(
            name="AnalyzerOutput",
            version=1,
            valid=False,
            errors=[str(e)],
        )
