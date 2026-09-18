"""Strongly typed domain objects shared by the offline research pipeline.

The objects deliberately keep vision confidence and temporal completeness separate.
That makes it impossible for an unfinished right-edge candle to silently become a
normal OHLC row.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


class SignalDirection(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    NO_TRADE = "NO_TRADE"


class Regime(str, Enum):
    TRENDING_BULL = "TRENDING_BULL"
    TRENDING_BEAR = "TRENDING_BEAR"
    RANGING = "RANGING"
    BREAKOUT = "BREAKOUT"
    POST_BREAKOUT = "POST_BREAKOUT"
    PULLBACK = "PULLBACK"
    REVERSAL_ATTEMPT = "REVERSAL_ATTEMPT"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    VOLATILITY_COMPRESSION = "VOLATILITY_COMPRESSION"
    CHOPPY = "CHOPPY"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


class VolatilityClass(str, Enum):
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ImageFrame:
    width: int
    height: int
    pixels: Tuple[Tuple[Tuple[int, int, int], ...], ...]
    source_format: str = "unknown"

    def pixel(self, x: int, y: int) -> Tuple[int, int, int]:
        return self.pixels[y][x]


@dataclass(frozen=True)
class Rect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class ScreenshotQualityReport:
    quality_score: float
    blur_score: float
    chart_coverage: float
    candle_visibility: float
    axis_visibility: float
    indicator_visibility: float
    occlusion_score: float
    resolution_adequacy: float
    duplicate: bool = False
    is_supported: bool = False
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class DetectedChart:
    bounds: Rect
    background_color: Tuple[int, int, int]
    bullish_color: Optional[Tuple[int, int, int]]
    bearish_color: Optional[Tuple[int, int, int]]
    grid_color: Optional[Tuple[int, int, int]]
    current_price_y: Optional[float]
    quality: ScreenshotQualityReport


@dataclass
class ReconstructedCandle:
    index: int
    x_center: float
    body_top: float
    body_bottom: float
    wick_top: float
    wick_bottom: float
    direction: Direction
    body_height: float
    upper_wick_length: float
    lower_wick_length: float
    total_range: float
    body_to_range_ratio: float
    relative_open: float
    relative_close: float
    relative_high: float
    relative_low: float
    is_complete: bool
    visual_confidence: float
    partial_observation: bool = False

    @property
    def midpoint(self) -> float:
        return (self.relative_open + self.relative_close) / 2.0


@dataclass
class CandleSequence:
    candles: List[ReconstructedCandle]
    chart: DetectedChart
    running_candle_index: Optional[int]
    reconstruction_confidence: float


@dataclass(frozen=True)
class SwingPoint:
    index: int
    value: float
    kind: str  # HIGH or LOW
    major: bool
    prominence: float


@dataclass
class MarketStructure:
    swing_points: List[SwingPoint]
    labels: List[str]
    bias: str
    break_of_structure: bool
    change_of_character: bool
    transition: bool
    score: float
    explanation: str


@dataclass
class SupportResistanceZone:
    kind: str
    low: float
    high: float
    strength: float
    touch_count: int
    recent_touch_weight: float
    rejection_strength: float
    break_count: int
    distance_from_current: float
    freshness: float
    confidence: float


@dataclass
class MomentumState:
    bullish_score: float
    bearish_score: float
    strength: float
    acceleration: float
    exhaustion_probability: float
    directional_efficiency: float


@dataclass
class VolatilityState:
    median_range: float
    range_std: float
    recent_to_long_range: float
    expansion: float
    contraction: float
    classification: VolatilityClass


@dataclass
class TrendState:
    short_slope: float
    visible_slope: float
    efficiency: float
    smoothness: float
    classification: str
    score: float


@dataclass
class MarketState:
    structure: MarketStructure
    zones: List[SupportResistanceZone]
    momentum: MomentumState
    volatility: VolatilityState
    trend: TrendState
    regime: Regime


@dataclass
class FeatureVector:
    values: List[float]
    names: List[str]
    schema_version: str
    valid_mask: List[bool]
    metadata: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        return dict(zip(self.names, self.values))


@dataclass
class ModelPrediction:
    model_name: str
    probability_up: float
    available: bool
    failure: Optional[str] = None


@dataclass
class UncertaintyEstimate:
    vision: float
    aleatoric: float
    epistemic: float
    disagreement: float
    distribution_shift: float
    combined: float


@dataclass
class SignalDecision:
    direction: SignalDirection
    calibrated_probability: Optional[float]
    confidence: Optional[float]
    abstained: bool
    abstention_reasons: List[str]
    market_regime: Regime
    bullish_evidence: List[str]
    bearish_evidence: List[str]
    neutral_evidence: List[str]
    model_agreement: Optional[float]
    uncertainty_score: Optional[float]
    screenshot_quality: float
    processing_time_ms: float
    model_version: str
    feature_schema_version: str
    model_predictions: List[ModelPrediction] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    timestamp_utc: str
    asset: Optional[str]
    screenshot_hash: str
    feature_vector: Tuple[float, ...]
    model_version: str
    prediction: SignalDirection
    probability: Optional[float]
    confidence: Optional[float]
    regime: Regime
    vision_quality: float
    actual_outcome: Optional[Direction] = None
    user_verified_outcome: Optional[Direction] = None
    notes: Optional[str] = None


@dataclass
class EvaluationReport:
    total_samples: int
    accepted_predictions: int
    abstentions: int
    coverage: float
    correct: int
    incorrect: int
    accuracy: Optional[float]
    balanced_accuracy: Optional[float]
    brier_score: Optional[float]
    log_loss: Optional[float]
    ece: Optional[float]
    longest_correct_streak: int
    longest_losing_streak: int
    by_confidence_bucket: Dict[str, Dict[str, float]]
    by_regime: Dict[str, Dict[str, float]]
    notes: List[str] = field(default_factory=list)
