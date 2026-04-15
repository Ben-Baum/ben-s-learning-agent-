"""
Rule-based Belief Graph Mapper — replaces the LLM-based Belief Graph Mapper.
Takes NLP extraction results and converts them to graph deltas using code rules.

This saves 1 API call per message by doing what the LLM Belief Graph Mapper did,
but with deterministic Python logic instead.
"""

from typing import Dict, Any, List
from models import (
    NLPExtractionResult,
    BeliefGraphDelta,
    GraphDeltaNode,
    GraphDeltaEdge,
    BeliefItem,
    EmotionItem,
    CognitiveDistortionItem,
    EmotionalCharge,
)


# ─── Edge relation mapping for distortion → belief connections ───
_DISTORTION_TO_RELATION = {
    "all_or_nothing": "explains",
    "catastrophizing": "causes",
    "mind_reading": "justifies",
    "fortune_telling": "causes",
    "overgeneralization": "explains",
    "labeling": "justifies",
    "emotional_reasoning": "explains",
    "should_statements": "justifies",
    "discounting_the_positive": "contradicts",
    "personalization": "causes",
    "other": "other",
}


def _make_belief_node(belief: BeliefItem) -> GraphDeltaNode:
    """Convert a BeliefItem from NLP extraction into a graph node."""
    return GraphDeltaNode(
        id=belief.id,
        statement=belief.statement,
        level=belief.level,
        valence=belief.valence,
        initial_strength=belief.strength,
        source_message_id=None,
    )


def _make_emotion_node(emotion: EmotionItem, idx: int) -> GraphDeltaNode:
    """Convert an EmotionItem into a graph node (for high-intensity emotions)."""
    node_id = f"emotion_{emotion.label}_{idx}"
    return GraphDeltaNode(
        id=node_id,
        statement=f"Feeling: {emotion.label} ({emotion.intensity:.1f})",
        level="self",
        valence="negative" if emotion.polarity == "pain" else (
            "positive" if emotion.polarity == "positive" else "neutral"
        ),
        initial_strength=emotion.intensity,
        source_message_id=None,
    )


def _link_beliefs_to_beliefs(beliefs: List[BeliefItem]) -> List[GraphDeltaEdge]:
    """Create edges between beliefs that share the same level or seem related."""
    edges = []
    for i, b1 in enumerate(beliefs):
        for b2 in beliefs[i + 1:]:
            # Same level beliefs support each other
            if b1.level == b2.level and b1.valence == b2.valence:
                edges.append(GraphDeltaEdge(
                    from_id=b1.id,
                    to_id=b2.id,
                    relation_type="supports",
                    weight_delta=0.3,
                ))
            # Contradicting valence
            elif b1.level == b2.level and (
                (b1.valence == "negative" and b2.valence == "positive")
                or (b1.valence == "positive" and b2.valence == "negative")
            ):
                edges.append(GraphDeltaEdge(
                    from_id=b1.id,
                    to_id=b2.id,
                    relation_type="contradicts",
                    weight_delta=0.2,
                ))
    return edges


def _link_distortions_to_beliefs(
    distortions: List[CognitiveDistortionItem],
    beliefs: List[BeliefItem],
) -> List[GraphDeltaEdge]:
    """Create edges from distortions to the beliefs they support/cause."""
    edges = []
    for d in distortions:
        relation = _DISTORTION_TO_RELATION.get(d.type, "other")
        # Link distortion to all negative beliefs (distortions typically feed negatives)
        for b in beliefs:
            if b.valence in ("negative", "mixed"):
                edges.append(GraphDeltaEdge(
                    from_id=f"distortion_{d.type}",
                    to_id=b.id,
                    relation_type=relation,
                    weight_delta=d.confidence * 0.4,
                ))
    return edges


def _link_emotions_to_beliefs(
    emotions: List[EmotionItem],
    beliefs: List[BeliefItem],
) -> List[GraphDeltaEdge]:
    """Link high-intensity emotions to beliefs they might be caused by."""
    edges = []
    for idx, e in enumerate(emotions):
        if e.intensity < 0.5:
            continue
        for b in beliefs:
            # Pain emotions link to negative beliefs
            if e.polarity == "pain" and b.valence in ("negative", "mixed"):
                edges.append(GraphDeltaEdge(
                    from_id=b.id,
                    to_id=f"emotion_{e.label}_{idx}",
                    relation_type="causes",
                    weight_delta=e.intensity * 0.3,
                ))
    return edges


def compute_belief_graph_delta_rules(
    nlp_result: NLPExtractionResult,
) -> BeliefGraphDelta:
    """
    Rule-based Belief Graph Delta computation.
    Replaces the LLM-based compute_belief_graph_delta.

    Rules:
    1. Every belief from NLP → becomes a graph node
    2. High-intensity emotions (>0.5) → become graph nodes
    3. Distortions with a type → create a placeholder node
    4. Same-level beliefs → "supports" edges
    5. Contradicting valence same-level → "contradicts" edges
    6. Distortions → linked to negative beliefs with appropriate relation
    7. High emotions → linked to negative beliefs via "causes"
    """
    nodes: List[GraphDeltaNode] = []
    edges: List[GraphDeltaEdge] = []

    # 1) Beliefs → nodes
    for belief in nlp_result.beliefs:
        nodes.append(_make_belief_node(belief))

    # 2) High-intensity emotions → nodes
    for idx, emotion in enumerate(nlp_result.emotions):
        if emotion.intensity >= 0.5:
            nodes.append(_make_emotion_node(emotion, idx))

    # 3) Distortions → placeholder nodes
    for d in nlp_result.cognitive_distortions:
        nodes.append(GraphDeltaNode(
            id=f"distortion_{d.type}",
            statement=f"Cognitive distortion: {d.type} — {d.explanation_short[:80]}",
            level="self",
            valence="negative",
            initial_strength=d.confidence * 0.5,
            source_message_id=None,
        ))

    # 4-7) Create edges
    edges.extend(_link_beliefs_to_beliefs(nlp_result.beliefs))
    edges.extend(_link_distortions_to_beliefs(
        nlp_result.cognitive_distortions, nlp_result.beliefs
    ))
    edges.extend(_link_emotions_to_beliefs(nlp_result.emotions, nlp_result.beliefs))

    # 8) Emotional charges → nodes (saved regardless of engagement_mode)
    for charge in nlp_result.emotional_charges:
        charge_node_id = f"charge_{charge.id}"
        nodes.append(GraphDeltaNode(
            id=charge_node_id,
            statement=f"[מטען] {charge.statement}",
            level="self",
            valence="negative",
            initial_strength=charge.intensity * (1.5 if charge.is_core else 1.0),
            source_message_id=None,
        ))
        # Link core charge to related beliefs
        if charge.is_core:
            for b in nlp_result.beliefs:
                if b.valence in ("negative", "mixed"):
                    edges.append(GraphDeltaEdge(
                        from_id=charge_node_id,
                        to_id=b.id,
                        relation_type="causes",
                        weight_delta=charge.intensity * 0.5,
                    ))

    return BeliefGraphDelta(
        schema_version="1.0",
        new_nodes=nodes,
        new_or_updated_edges=edges,
    )
