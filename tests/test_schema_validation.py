"""Schema parsing and fallback when LLM returns bad JSON."""

import json
import pytest
from genie.schemas import AnalyzerOutput, ControlLayer
from genie.parser import parse_analyzer_output, strip_json_block


def test_parse_valid_analyzer_output():
    valid = {
        "version": 1,
        "graph_delta": {"upsert_nodes": [], "upsert_edges": [], "deprecate_node_ids": [], "deprecate_edge_ids": []},
        "control_layer": {
            "recommended_mode": "explore",
            "som_mode_recommendation": "hold",
            "clarity_score": 0.5,
            "readiness_score": 0.4,
            "resistance_detected": False,
            "resistance_level": None,
            "resistance_hypothesis": None,
            "allowed_actions": [],
            "block_change_agent": False,
            "active_hubs": [],
            "primary_loop_signature": None,
            "next_questions": [],
            "memory_policy": {"surface_personal_memory": "never", "surface_situational_memory": "never"},
            "safety_flag": False,
            "safety_reason": None,
        },
        "memory_candidates": [],
    }
    raw = json.dumps(valid)
    out, schema = parse_analyzer_output(raw)
    assert out is not None
    assert schema.valid is True
    assert out.version == 1
    assert out.control_layer.recommended_mode == "explore"
    assert out.control_layer.safety_flag is False


def test_parse_invalid_json_marks_schema_invalid():
    out, schema = parse_analyzer_output("not json at all")
    assert out is None
    assert schema.valid is False
    assert len(schema.errors) >= 1


def test_parse_malformed_json_fallback():
    out, schema = parse_analyzer_output('{"version": 1, "graph_delta": invalid}')
    assert out is None
    assert schema.valid is False


def test_strip_json_block():
    assert strip_json_block('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_json_block('{"a":1}') == '{"a":1}'


def test_fallback_implies_stabilize():
    """When schema is invalid, orchestrator should use stabilize (tested via parse returning valid=False)."""
    out, schema = parse_analyzer_output("nope")
    assert schema.valid is False
    # Caller is expected to set mode=stabilize, block_change_agent=True when schema.valid is False
    assert out is None
