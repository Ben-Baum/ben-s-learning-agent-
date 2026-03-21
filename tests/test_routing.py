"""Orchestrator routing rules match spec."""

import pytest
from genie.schemas import AnalyzerOutput, ControlLayer, GraphDelta
from genie.orchestrator import decide_routing


def _analyzer(safety_flag=False, resistance_detected=False, readiness_score=0.0, block_change_agent=False):
    return AnalyzerOutput(
        version=1,
        graph_delta=GraphDelta(),
        control_layer=ControlLayer(
            safety_flag=safety_flag,
            resistance_detected=resistance_detected,
            readiness_score=readiness_score,
            block_change_agent=block_change_agent,
        ),
    )


def test_safety_flag_implies_stabilize_and_block():
    a = _analyzer(safety_flag=True)
    r = decide_routing(a)
    assert r.mode == "stabilize"
    assert r.block_change_agent is True


def test_resistance_detected_implies_explore_resistance_and_block():
    a = _analyzer(resistance_detected=True, safety_flag=False)
    r = decide_routing(a)
    assert r.mode == "explore_resistance"
    assert r.block_change_agent is True


def test_readiness_ge_065_implies_intervene():
    a = _analyzer(readiness_score=0.65, resistance_detected=False, safety_flag=False)
    r = decide_routing(a)
    assert r.mode == "intervene"


def test_readiness_above_065_intervene():
    a = _analyzer(readiness_score=0.9, resistance_detected=False, safety_flag=False)
    r = decide_routing(a)
    assert r.mode == "intervene"


def test_else_explore():
    a = _analyzer(readiness_score=0.5, resistance_detected=False, safety_flag=False)
    r = decide_routing(a)
    assert r.mode == "explore"


def test_safety_takes_precedence_over_resistance():
    a = _analyzer(safety_flag=True, resistance_detected=True)
    r = decide_routing(a)
    assert r.mode == "stabilize"
