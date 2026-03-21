"""
graph_utils.py — Belief Graph utility functions

Applies a BeliefGraphDelta onto the current belief graph JSON dict.
"""

from __future__ import annotations
from typing import Any, Dict


def apply_delta_to_graph(
    current_graph_json: Dict[str, Any],
    delta,  # BeliefGraphDelta
) -> Dict[str, Any]:
    """
    Merge a BeliefGraphDelta into the current graph dict and return updated graph.

    - New nodes are added (or existing nodes have their strength accumulated).
    - Edges are added or have their weight updated (weight_delta is accumulated).
    """
    # Deep-copy to avoid mutating caller's dict
    import copy
    graph = copy.deepcopy(current_graph_json)

    # Ensure structure exists
    if "nodes" not in graph:
        graph["nodes"] = {}
    if "edges" not in graph:
        graph["edges"] = []
    if "schema_version" not in graph:
        graph["schema_version"] = "1.0"

    # ── Apply new/updated nodes ──────────────────────────────────────────────
    for node in delta.new_nodes:
        if node.id in graph["nodes"]:
            # Accumulate strength (cap at 1.0)
            existing = graph["nodes"][node.id]
            existing["strength"] = min(1.0, existing.get("strength", 0.0) + node.initial_strength)
        else:
            graph["nodes"][node.id] = {
                "id": node.id,
                "statement": node.statement,
                "level": node.level,
                "valence": node.valence,
                "strength": node.initial_strength,
            }

    # ── Apply new/updated edges ──────────────────────────────────────────────
    for edge in delta.new_or_updated_edges:
        # Look for existing edge with same from/to/relation
        found = False
        for existing_edge in graph["edges"]:
            if (
                existing_edge.get("from_id") == edge.from_id
                and existing_edge.get("to_id") == edge.to_id
                and existing_edge.get("relation_type") == edge.relation_type
            ):
                # Accumulate weight, clamped to [-1.0, 1.0]
                existing_edge["weight"] = max(
                    -1.0,
                    min(1.0, existing_edge.get("weight", 0.0) + edge.weight_delta),
                )
                found = True
                break

        if not found:
            graph["edges"].append({
                "from_id": edge.from_id,
                "to_id": edge.to_id,
                "relation_type": edge.relation_type,
                "weight": max(-1.0, min(1.0, edge.weight_delta)),
            })

    return graph
