from typing import List, Literal, Optional, Dict
from pydantic import BaseModel, field_validator

_VALID_EMOTION_LABELS = {
    "sadness", "anxiety", "fear", "shame", "guilt", "anger",
    "frustration", "relief", "hope", "confusion", "numbness", "other",
}

_VALID_POLARITY = {"pain", "coping", "positive", "neutral"}


# ========= 1) NLP Extraction =========


class EmotionItem(BaseModel):
    label: str
    intensity: float  # 0.0–1.0
    evidence_snippet: str
    polarity: str
    confidence: float  # 0.0–1.0

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, v: str) -> str:
        return v if v in _VALID_EMOTION_LABELS else "other"

    @field_validator("polarity", mode="before")
    @classmethod
    def normalize_polarity(cls, v: str) -> str:
        return v if v in _VALID_POLARITY else "neutral"


_VALID_DISTORTION_TYPES = {
    "all_or_nothing", "catastrophizing", "mind_reading", "fortune_telling",
    "overgeneralization", "labeling", "emotional_reasoning", "should_statements",
    "discounting_the_positive", "personalization", "other",
}

class CognitiveDistortionItem(BaseModel):
    type: str
    evidence_snippet: str
    explanation_short: str  # technical, not for user
    confidence: float  # 0.0–1.0

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: str) -> str:
        return v if v in _VALID_DISTORTION_TYPES else "other"


class BeliefItem(BaseModel):
    id: str
    level: Literal["self", "others", "world", "future"]
    valence: Literal["negative", "neutral", "positive", "mixed"]
    statement: str
    evidence_snippet: Optional[str] = None
    strength: float  # 0.0–1.0 for this message
    confidence: float  # 0.0–1.0


class MetaInfo(BaseModel):
    language: Literal["he", "en", "mixed", "other"]
    message_length_chars: int
    message_id: Optional[str] = None
    turn_index: Optional[int] = None


class NLPExtractionResult(BaseModel):
    schema_version: str  # e.g. "1.0"
    raw_text: str
    emotions: List[EmotionItem]
    cognitive_distortions: List[CognitiveDistortionItem]
    beliefs: List[BeliefItem]
    meta: MetaInfo


# ========= 2) Belief Graph & Delta =========


class GraphDeltaNode(BaseModel):
    id: str
    statement: str
    level: Literal["self", "others", "world", "future"]
    valence: Literal["negative", "neutral", "positive", "mixed"]
    initial_strength: float  # 0.0–1.0
    source_message_id: Optional[str] = None


class GraphDeltaEdge(BaseModel):
    from_id: str
    to_id: str
    relation_type: Literal[
        "causes",
        "explains",
        "justifies",
        "contradicts",
        "supports",
        "coping_strategy",
        "other",
    ]
    weight_delta: float  # -1.0–1.0
    source_message_id: Optional[str] = None


class BeliefGraphDelta(BaseModel):
    schema_version: str  # e.g. "1.0"
    new_nodes: List[GraphDeltaNode]
    new_or_updated_edges: List[GraphDeltaEdge]


class BeliefNode(BaseModel):
    id: str
    statement: str
    level: Literal["self", "others", "world", "future"]
    valence: Literal["negative", "neutral", "positive", "mixed"]
    strength: float  # 0.0–1.0, accumulated over time


class BeliefEdge(BaseModel):
    from_id: str
    to_id: str
    relation_type: Literal[
        "causes",
        "explains",
        "justifies",
        "contradicts",
        "supports",
        "coping_strategy",
        "other",
    ]
    weight: float  # -1.0–1.0, accumulated


class BeliefGraph(BaseModel):
    schema_version: str = "1.0"
    nodes: Dict[str, BeliefNode] = {}
    edges: List[BeliefEdge] = []


# ========= 3) Tactical Strategy / Investigation Vectors =========


class InvestigationVector(BaseModel):
    id: str  # e.g. "explore_resistance_to_change"
    priority: Literal["high", "medium", "low"]
    focus_type: Literal[
        "resistance",
        "emotion_clarification",
        "context_clarification",
        "values",
        "identity",
        "coping_strategies",
        "relationships",
        "future_fears",
        "other",
    ]
    short_description: str  # internal description, not for user
    suggested_angle_for_front_agent: str  # natural language hint


class StrategyMeta(BaseModel):
    schema_version: str  # e.g. "1.0"
    detected_resistance: bool
    strongest_signal_belief_ids: List[str]
    notes_technical: Optional[str] = None


class TacticalStrategyResult(BaseModel):
    meta: StrategyMeta
    investigation_vectors: List[InvestigationVector]

