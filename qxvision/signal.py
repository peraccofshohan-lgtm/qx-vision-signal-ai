"""Uncertainty-aware abstention policy. NO_TRADE is the safe default."""
from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .domain import *
from .model import Ensemble


class SignalPolicy:
    def __init__(self, min_probability: float = 0.64, min_quality: float = 0.52, min_agreement: float = 0.62, max_uncertainty: float = 0.42, min_candles: int = 8):
        if not 0.5 < min_probability < 1.0: raise ValueError("min_probability must be > .5 and < 1")
        self.min_probability, self.min_quality, self.min_agreement, self.max_uncertainty, self.min_candles = min_probability, min_quality, min_agreement, max_uncertainty, min_candles


def _uncertainty(quality: float, reconstruction: float, predictions: Sequence[ModelPrediction], agreement: Optional[float], ood_score: Optional[float], state: MarketState) -> UncertaintyEstimate:
    vision = 1.0 - max(0.0, min(1.0, quality * reconstruction))
    ps = [p.probability_up for p in predictions if p.available]
    epistemic = min(1.0, math.sqrt(sum((p - (sum(ps) / len(ps))) ** 2 for p in ps) / max(1, len(ps))) * 3) if ps else 1.0
    aleatoric = min(1.0, 1.0 - max(state.momentum.strength, state.trend.score) * 0.55 + state.volatility.expansion * 0.45)
    disagreement = 1.0 - agreement if agreement is not None else 1.0
    shift = min(1.0, (ood_score or 0.0) / 8.0)
    combined = min(1.0, 0.35 * vision + 0.20 * aleatoric + 0.25 * epistemic + 0.15 * disagreement + 0.05 * shift)
    return UncertaintyEstimate(vision, aleatoric, epistemic, disagreement, shift, combined)


def _evidence(state: MarketState) -> tuple[List[str], List[str], List[str]]:
    bull: List[str] = []; bear: List[str] = []; neutral: List[str] = []
    if state.trend.classification in ("BULLISH", "STRONG_BULLISH", "WEAK_BULLISH"): bull.append(f"trend {state.trend.classification.lower()}")
    elif state.trend.classification in ("BEARISH", "STRONG_BEARISH", "WEAK_BEARISH"): bear.append(f"trend {state.trend.classification.lower()}")
    else: neutral.append("trend is range or transition")
    if state.momentum.bullish_score > state.momentum.bearish_score + 0.12: bull.append("recent body momentum favors up")
    elif state.momentum.bearish_score > state.momentum.bullish_score + 0.12: bear.append("recent body momentum favors down")
    else: neutral.append("momentum is mixed")
    if state.structure.bias == "BULLISH": bull.append("higher-high/higher-low structure")
    elif state.structure.bias == "BEARISH": bear.append("lower-high/lower-low structure")
    else: neutral.append("structure is ambiguous")
    if state.volatility.classification in (VolatilityClass.HIGH, VolatilityClass.EXTREME): neutral.append(f"volatility {state.volatility.classification.value.lower()}")
    return bull, bear, neutral


def decide_with_features(candles: Sequence[ReconstructedCandle], quality: ScreenshotQualityReport, state: MarketState, vector: FeatureVector, ensemble: Optional[Ensemble], *, policy: SignalPolicy = SignalPolicy(), processing_time_ms: float = 0.0, model_version: str = "v1-untrained", ood_score: Optional[float] = None) -> SignalDecision:
    """Final entry point; the feature vector is explicit to prevent hidden recomputation."""
    bull, bear, neutral = _evidence(state); reasons: List[str] = []
    if not quality.is_supported: reasons.extend(quality.failure_reasons or ["SCREENSHOT_NOT_SUPPORTED"])
    if quality.quality_score < policy.min_quality: reasons.append("SCREENSHOT_QUALITY_INSUFFICIENT")
    if len(candles) < policy.min_candles: reasons.append("TOO_FEW_RELIABLE_CANDLES")
    if state.volatility.classification == VolatilityClass.EXTREME: reasons.append("EXTREME_VOLATILITY")
    if candles and candles[-1].partial_observation and candles[-1].visual_confidence < 0.35: reasons.append("RUNNING_CANDLE_UNSTABLE")
    if state.structure.bias == "NEUTRAL": reasons.append("AMBIGUOUS_STRUCTURE")
    if ensemble is None: reasons.append("MODEL_ARTIFACT_UNAVAILABLE")
    preds: List[ModelPrediction] = []; calibrated = None; agreement = None
    if ensemble is not None:
        preds, calibrated, agreement, failure = ensemble.predict(vector, state)
        if failure: reasons.append(failure)
    uncertainty = _uncertainty(quality.quality_score, sum(c.visual_confidence for c in candles) / max(1, len(candles)), preds, agreement, ood_score, state)
    if agreement is None or agreement < policy.min_agreement: reasons.append("LOW_MODEL_AGREEMENT")
    if uncertainty.combined > policy.max_uncertainty: reasons.append("UNCERTAINTY_TOO_HIGH")
    if calibrated is None: reasons.append("CALIBRATED_PROBABILITY_UNAVAILABLE")
    elif max(calibrated, 1 - calibrated) < policy.min_probability: reasons.append("PROBABILITY_BELOW_POLICY_THRESHOLD")
    direction = SignalDirection.NO_TRADE; abstained = True
    confidence = max(calibrated, 1 - calibrated) if calibrated is not None else None
    if not reasons and calibrated is not None:
        direction = SignalDirection.UP if calibrated >= 0.5 else SignalDirection.DOWN; abstained = False
    return SignalDecision(direction, calibrated, confidence, abstained, sorted(set(reasons)), state.regime, bull, bear, neutral, agreement, uncertainty.combined, quality.quality_score, processing_time_ms, model_version, vector.schema_version, preds)
