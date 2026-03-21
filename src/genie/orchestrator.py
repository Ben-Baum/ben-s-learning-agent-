"""Deterministic orchestrator: resistance-first state machine and routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from genie.schemas import AnalyzerOutput, ControlLayer

SystemMode = Literal["explore", "explore_resistance", "intervene", "stabilize", "renewal"]
StyleHint = Literal["engaging"]  # "Engaging mode" for Front style hint only


@dataclass
class RoutingDecision:
    """Result of orchestrator decision (non-LLM)."""
    mode: SystemMode
    block_change_agent: bool
    style_hint: StyleHint = "engaging"


def decide_routing(analyzer_output: AnalyzerOutput) -> RoutingDecision:
    """Deterministic rules (MUST):
    - If safety_flag → mode=stabilize, block_change_agent=True
    - Else if resistance_detected → mode=explore_resistance, block_change_agent=True
    - Else if readiness_score >= 0.65 → mode=intervene
    - Else → mode=explore
    """
    cl = analyzer_output.control_layer
    if cl.safety_flag:
        return RoutingDecision(mode="stabilize", block_change_agent=True)
    if cl.resistance_detected:
        return RoutingDecision(mode="explore_resistance", block_change_agent=True)
    if cl.readiness_score >= 0.65:
        return RoutingDecision(mode="intervene", block_change_agent=cl.block_change_agent)
    return RoutingDecision(mode="explore", block_change_agent=cl.block_change_agent)


def get_control_layer_for_front(analyzer_output: AnalyzerOutput, decision: RoutingDecision) -> ControlLayer:
    """Return control layer with orchestrator-overridden mode for Front agent."""
    cl = analyzer_output.control_layer.model_copy(deep=True)
    cl.recommended_mode = decision.mode
    cl.block_change_agent = decision.block_change_agent
    return cl
