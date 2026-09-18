"""Deterministic technical feature engines over reconstructed normalized candles."""
from __future__ import annotations

import math
import statistics
from typing import Iterable, List, Sequence, Tuple

from .domain import *

FEATURE_SCHEMA_VERSION = "qxvision.features.v1"


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _close(c: ReconstructedCandle) -> float:
    return c.relative_close


def _range(c: ReconstructedCandle) -> float:
    return max(1e-6, c.total_range)


def _direction_value(c: ReconstructedCandle) -> float:
    return 1.0 if c.direction == Direction.UP else -1.0 if c.direction == Direction.DOWN else 0.0


def detect_swings(candles: Sequence[ReconstructedCandle]) -> List[SwingPoint]:
    usable = list(candles)
    if len(usable) < 3: return []
    ranges = [_range(c) for c in usable]
    threshold = max(0.002, statistics.median(ranges) * 0.35)
    points: List[SwingPoint] = []
    for i in range(1, len(usable) - 1):
        c = usable[i]
        hi = c.relative_high
        lo = c.relative_low
        left_hi, right_hi = usable[i - 1].relative_high, usable[i + 1].relative_high
        left_lo, right_lo = usable[i - 1].relative_low, usable[i + 1].relative_low
        high_prom = min(hi - left_hi, hi - right_hi)
        low_prom = min(left_lo - lo, right_lo - lo)
        if high_prom > threshold:
            points.append(SwingPoint(i, hi, "HIGH", high_prom >= threshold * 2.0, high_prom))
        if low_prom > threshold:
            points.append(SwingPoint(i, lo, "LOW", low_prom >= threshold * 2.0, low_prom))
    return points


def market_structure(candles: Sequence[ReconstructedCandle]) -> MarketStructure:
    swings = detect_swings(candles)
    highs = [s for s in swings if s.kind == "HIGH"]
    lows = [s for s in swings if s.kind == "LOW"]
    labels: List[str] = []
    for seq in (highs, lows):
        for previous, current in zip(seq, seq[1:]):
            if current.kind == "HIGH": labels.append("HH" if current.value > previous.value else "LH")
            else: labels.append("HL" if current.value > previous.value else "LL")
    bullish = sum(1 for x in labels if x in ("HH", "HL"))
    bearish = sum(1 for x in labels if x in ("LH", "LL"))
    bias = "BULLISH" if bullish > bearish + 1 else "BEARISH" if bearish > bullish + 1 else "NEUTRAL"
    close = _close(candles[-1]) if candles else 0.0
    prev_high = max((s.value for s in highs if s.index < len(candles) - 2), default=close)
    prev_low = min((s.value for s in lows if s.index < len(candles) - 2), default=close)
    bos_up = close > prev_high + 0.5 * (statistics.median([_range(c) for c in candles]) if candles else 0)
    bos_down = close < prev_low - 0.5 * (statistics.median([_range(c) for c in candles]) if candles else 0)
    bos = bos_up or bos_down
    transition = len(labels) >= 2 and labels[-1] != labels[-2]
    choch = transition and (labels[-1] in ("HH", "HL")) != (labels[-2] in ("HH", "HL"))
    score = _clamp(abs(bullish - bearish) / max(1.0, len(labels)))
    exp = f"{len(swings)} adaptive swings; {bias.lower()} sequence ({bullish} bullish vs {bearish} bearish labels)"
    if bos: exp += "; possible swing break"
    if choch: exp += "; transition detected"
    return MarketStructure(swings, labels, bias, bos, choch, transition, score, exp)


def support_resistance(candles: Sequence[ReconstructedCandle], max_zones: int = 6) -> List[SupportResistanceZone]:
    if not candles: return []
    values = []
    for c in candles:
        values.extend([(c.relative_high, "RESISTANCE", c.upper_wick_length), (c.relative_low, "SUPPORT", c.lower_wick_length), (c.midpoint, "BODY", c.body_height)])
    scale = max(0.004, statistics.median([_range(c) for c in candles]) * 0.65)
    clusters: List[dict] = []
    for value, kind, rejection in values:
        match = next((z for z in clusters if z["kind"] == kind and abs(value - z["center"]) <= scale), None)
        if match is None:
            clusters.append({"kind": "SUPPORT" if kind == "SUPPORT" else "RESISTANCE", "center": value, "values": [value], "rejections": [rejection], "indices": [len(clusters)]})
        else:
            match["values"].append(value); match["rejections"].append(rejection)
            match["center"] = sum(match["values"]) / len(match["values"])
    current = candles[-1].midpoint
    zones: List[SupportResistanceZone] = []
    for z in clusters:
        vals = z["values"]
        low, high = min(vals) - scale * 0.35, max(vals) + scale * 0.35
        touches = len(vals)
        recent = sum(1.0 + i / max(1, len(candles)) for i in range(max(0, len(candles) - touches), len(candles))) / max(1, touches)
        reject = _clamp(sum(z["rejections"]) / max(1, touches) / max(scale, 1e-6))
        breaks = sum(1 for c in candles if (c.relative_close > high if z["kind"] == "RESISTANCE" else c.relative_close < low))
        strength = _clamp(0.12 * touches + 0.45 * reject + 0.25 * min(1.0, recent / 2.0) - 0.06 * min(4, breaks))
        zones.append(SupportResistanceZone(z["kind"], low, high, strength, touches, recent, reject, breaks, abs(current - z["center"]), _clamp(1.0 - breaks / max(1, touches + breaks)), strength))
    zones.sort(key=lambda x: x.strength, reverse=True)
    return zones[:max_zones]


def momentum(candles: Sequence[ReconstructedCandle]) -> MomentumState:
    if not candles: return MomentumState(0, 0, 0, 0, 1, 0)
    complete = list(candles[:-1]) if not candles[-1].is_complete else list(candles)
    tail = complete[-8:] or complete
    signed = [_direction_value(c) * c.body_to_range_ratio for c in tail]
    weighted = sum((i + 1) * v for i, v in enumerate(signed)) / max(1, sum(range(1, len(signed) + 1)))
    bull = _clamp((weighted + 1) / 2); bear = _clamp((1 - weighted) / 2)
    strength = _clamp(abs(weighted))
    if len(signed) >= 4:
        acceleration = sum(signed[-2:]) / 2 - sum(signed[:2]) / 2
    else: acceleration = 0.0
    same = sum(1 for a, b in zip(signed, signed[1:]) if a * b > 0)
    exhaustion = _clamp((sum(c.upper_wick_length + c.lower_wick_length for c in tail[-3:]) / max(1, len(tail[-3:]))) * 3 + max(0, abs(weighted) - 0.7) * 0.8)
    efficiency = abs(sum(signed)) / max(1e-6, sum(abs(x) for x in signed))
    return MomentumState(bull, bear, strength, _clamp((acceleration + 1) / 2), exhaustion, _clamp(efficiency))


def volatility(candles: Sequence[ReconstructedCandle]) -> VolatilityState:
    ranges = [_range(c) for c in candles[:-1] if c.is_complete] or [_range(c) for c in candles]
    if not ranges: return VolatilityState(0, 0, 0, 0, 0, VolatilityClass.UNKNOWN)
    med = statistics.median(ranges); std = statistics.pstdev(ranges) if len(ranges) > 1 else 0.0
    recent = statistics.median(ranges[-5:])
    ratio = recent / max(med, 1e-6)
    expansion = _clamp((ratio - 1) / 1.5); contraction = _clamp((1 - ratio) / 0.7)
    classification = VolatilityClass.VERY_LOW if ratio < 0.45 else VolatilityClass.LOW if ratio < 0.75 else VolatilityClass.NORMAL if ratio < 1.35 else VolatilityClass.HIGH if ratio < 2.2 else VolatilityClass.EXTREME
    return VolatilityState(med, std, ratio, expansion, contraction, classification)


def trend(candles: Sequence[ReconstructedCandle]) -> TrendState:
    closes = [_close(c) for c in candles[:-1] if c.is_complete] or [_close(c) for c in candles]
    if len(closes) < 2: return TrendState(0, 0, 0, 0, "TRANSITION", 0)
    def slope(xs: Sequence[float]) -> float:
        n = len(xs); mx = (n - 1) / 2; my = sum(xs) / n
        den = sum((i - mx) ** 2 for i in range(n)) or 1
        return sum((i - mx) * (v - my) for i, v in enumerate(xs)) / den
    short = slope(closes[-min(6, len(closes)):]); visible = slope(closes)
    changes = [b - a for a, b in zip(closes, closes[1:])]
    efficiency = abs(closes[-1] - closes[0]) / max(1e-6, sum(abs(x) for x in changes))
    smooth = 1.0 - min(1.0, (statistics.pstdev(changes) if len(changes) > 1 else 0) / max(1e-6, statistics.mean(abs(x) for x in changes) if changes else 1))
    score = _clamp(abs(short) * 12 + efficiency * 0.5)
    if score < 0.18: cls = "RANGE"
    elif short > 0 and score > 0.72: cls = "STRONG_BULLISH"
    elif short > 0: cls = "BULLISH" if score > 0.38 else "WEAK_BULLISH"
    elif score > 0.72: cls = "STRONG_BEARISH"
    else: cls = "BEARISH" if score > 0.38 else "WEAK_BEARISH"
    return TrendState(short, visible, _clamp(efficiency), _clamp(smooth), cls, score)


def regime(state: MarketState | None = None, *, structure: MarketStructure | None = None, zones: Sequence[SupportResistanceZone] = (), mom: MomentumState | None = None, vol: VolatilityState | None = None, tr: TrendState | None = None) -> Regime:
    if state is not None:
        structure, zones, mom, vol, tr = state.structure, state.zones, state.momentum, state.volatility, state.trend
    if not all((structure, mom, vol, tr)): return Regime.UNKNOWN
    assert structure and mom and vol and tr
    if vol.classification == VolatilityClass.EXTREME: return Regime.VOLATILITY_EXPANSION
    if vol.classification == VolatilityClass.VERY_LOW: return Regime.VOLATILITY_COMPRESSION
    if structure.change_of_character or structure.transition: return Regime.TRANSITION
    if structure.break_of_structure and tr.score > 0.45: return Regime.BREAKOUT
    if tr.classification in ("STRONG_BULLISH", "BULLISH"): return Regime.TRENDING_BULL
    if tr.classification in ("STRONG_BEARISH", "BEARISH"): return Regime.TRENDING_BEAR
    if mom.exhaustion_probability > 0.7: return Regime.REVERSAL_ATTEMPT
    if tr.classification == "RANGE": return Regime.RANGING
    if tr.efficiency < 0.25: return Regime.CHOPPY
    return Regime.TRANSITION


def morphology(candles: Sequence[ReconstructedCandle]) -> dict:
    if not candles: return {"doji": 0.0, "hammer": 0.0, "shooting_star": 0.0, "engulfing_bull": 0.0, "engulfing_bear": 0.0, "inside": 0.0, "marubozu": 0.0}
    c = candles[-1]; p = candles[-2] if len(candles) > 1 else None
    small = 1.0 if c.body_to_range_ratio <= 0.18 else 0.0
    hammer = 1.0 if c.lower_wick_length > c.body_height * 1.8 and c.upper_wick_length < c.body_height else 0.0
    star = 1.0 if c.upper_wick_length > c.body_height * 1.8 and c.lower_wick_length < c.body_height else 0.0
    bull_engulf = 1.0 if p and c.direction == Direction.UP and p.direction == Direction.DOWN and c.body_height > p.body_height * 1.05 else 0.0
    bear_engulf = 1.0 if p and c.direction == Direction.DOWN and p.direction == Direction.UP and c.body_height > p.body_height * 1.05 else 0.0
    inside = 1.0 if p and c.relative_high <= p.relative_high and c.relative_low >= p.relative_low else 0.0
    marubozu = 1.0 if c.body_to_range_ratio > 0.82 else 0.0
    return {"doji": small, "hammer": hammer, "shooting_star": star, "engulfing_bull": bull_engulf, "engulfing_bear": bear_engulf, "inside": inside, "marubozu": marubozu}


def build_state(candles: Sequence[ReconstructedCandle]) -> MarketState:
    # Structure, zones, trend, momentum and volatility use completed bars only.
    # The running bar remains available to the feature vector under its explicit
    # running_* names.
    usable = list(candles[:-1]) if candles and not candles[-1].is_complete else list(candles)
    usable = usable or list(candles)
    s = market_structure(usable); z = support_resistance(usable); m = momentum(usable); v = volatility(usable); t = trend(usable)
    return MarketState(s, z, m, v, t, regime(structure=s, zones=z, mom=m, vol=v, tr=t))


def build_feature_vector(candles: Sequence[ReconstructedCandle], quality: ScreenshotQualityReport, state: MarketState) -> FeatureVector:
    # Running-candle values are deliberately under a named group and scaled by
    # its visual confidence. Training code can exclude them for a strict model.
    c = candles[-1] if candles else None
    complete = list(candles[:-1]) if c and not c.is_complete else list(candles)
    tail = complete[-8:]
    ret = [_close(b) - _close(a) for a, b in zip(tail, tail[1:])]
    names = [
        "quality_score", "blur_score", "chart_coverage", "candle_visibility", "reconstruction_confidence",
        "last_body_ratio", "last_upper_wick", "last_lower_wick", "last_range", "last_direction",
        "return_1", "return_2", "return_3", "return_mean", "body_mean", "range_mean",
        "structure_score", "structure_bullish", "structure_bearish", "bos", "choch",
        "short_slope", "visible_slope", "trend_efficiency", "trend_score", "momentum_bull", "momentum_bear", "momentum_strength", "momentum_acceleration", "exhaustion", "directional_efficiency",
        "median_range", "range_std", "recent_long_ratio", "vol_expansion", "vol_contraction",
        "is_vol_extreme", "is_range", "is_trending_bull", "is_trending_bear", "is_transition",
        "nearest_support_distance", "nearest_resistance_distance", "support_strength", "resistance_strength",
        "doji", "hammer", "shooting_star", "engulfing_bull", "engulfing_bear", "inside", "marubozu",
        "running_present", "running_direction", "running_body_ratio", "running_range", "running_confidence",
    ]
    vals = [quality.quality_score, quality.blur_score, quality.chart_coverage, quality.candle_visibility, sum(x.visual_confidence for x in candles) / max(1, len(candles))]
    if c:
        vals += [c.body_to_range_ratio, c.upper_wick_length, c.lower_wick_length, c.total_range, _direction_value(c)]
    else: vals += [0.0] * 5
    vals += [(ret[-1] if len(ret) >= 1 else 0), (ret[-2] if len(ret) >= 2 else 0), (ret[-3] if len(ret) >= 3 else 0), sum(ret) / max(1, len(ret)), statistics.mean([x.body_height for x in tail]) if tail else 0, statistics.mean([x.total_range for x in tail]) if tail else 0]
    vals += [state.structure.score, 1.0 if state.structure.bias == "BULLISH" else 0.0, 1.0 if state.structure.bias == "BEARISH" else 0.0, float(state.structure.break_of_structure), float(state.structure.change_of_character)]
    vals += [state.trend.short_slope, state.trend.visible_slope, state.trend.efficiency, state.trend.score, state.momentum.bullish_score, state.momentum.bearish_score, state.momentum.strength, state.momentum.acceleration, state.momentum.exhaustion_probability, state.momentum.directional_efficiency]
    vals += [state.volatility.median_range, state.volatility.range_std, state.volatility.recent_to_long_range, state.volatility.expansion, state.volatility.contraction, float(state.volatility.classification == VolatilityClass.EXTREME), float(state.regime == Regime.RANGING), float(state.regime == Regime.TRENDING_BULL), float(state.regime == Regime.TRENDING_BEAR), float(state.regime == Regime.TRANSITION)]
    supports = [z for z in state.zones if z.kind == "SUPPORT"]; resistances = [z for z in state.zones if z.kind == "RESISTANCE"]
    vals += [(min(s.distance_from_current for s in supports) if supports else 1.0), (min(s.distance_from_current for s in resistances) if resistances else 1.0), (max((s.strength for s in supports), default=0)), (max((s.strength for s in resistances), default=0))]
    vals += list(morphology(candles).values())
    vals += [float(bool(c and not c.is_complete)), _direction_value(c) if c and not c.is_complete else 0.0, c.body_to_range_ratio if c and not c.is_complete else 0.0, c.total_range if c and not c.is_complete else 0.0, c.visual_confidence if c and not c.is_complete else 0.0]
    # Protect a stable schema from accidental float NaN/inf propagation.
    vals = [0.0 if not math.isfinite(v) else float(v) for v in vals]
    return FeatureVector(vals, names, FEATURE_SCHEMA_VERSION, [math.isfinite(v) for v in vals], {"regime": state.regime.value, "running_candle_excluded_from_target": "true"})
