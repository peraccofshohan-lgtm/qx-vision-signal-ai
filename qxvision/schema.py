"""Single source of truth for the versioned 57-feature contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .features import FEATURE_SCHEMA_VERSION


def _description(name: str) -> str:
    group = "running-candle observation" if name.startswith("running_") else "screenshot quality" if name in {"quality_score", "blur_score", "chart_coverage", "candle_visibility", "reconstruction_confidence"} else "candle geometry" if name in {"last_body_ratio", "last_upper_wick", "last_lower_wick", "last_range", "last_direction"} else "recent sequence return" if name.startswith("return_") else "market structure" if name in {"structure_score", "structure_bullish", "structure_bearish", "bos", "choch"} else "trend and momentum" if name in {"short_slope", "visible_slope", "trend_efficiency", "trend_score", "momentum_bull", "momentum_bear", "momentum_strength", "momentum_acceleration", "exhaustion", "directional_efficiency"} else "volatility and regime" if name in {"median_range", "range_std", "recent_long_ratio", "vol_expansion", "vol_contraction", "is_vol_extreme", "is_range", "is_trending_bull", "is_trending_bear", "is_transition"} else "support/resistance zone" if name in {"nearest_support_distance", "nearest_resistance_distance", "support_strength", "resistance_strength"} else "candlestick morphology"
    return f"{name}: normalized {group} measurement"


def feature_schema() -> Dict[str, Any]:
    # Importing a tiny deterministic synthetic fixture would couple schema
    # generation to image decoding. Names are declared here and checked against
    # build_feature_vector in tests.
    names = [
        "quality_score", "blur_score", "chart_coverage", "candle_visibility", "reconstruction_confidence",
        "last_body_ratio", "last_upper_wick", "last_lower_wick", "last_range", "last_direction",
        "return_1", "return_2", "return_3", "return_mean", "body_mean", "range_mean",
        "structure_score", "structure_bullish", "structure_bearish", "bos", "choch",
        "short_slope", "visible_slope", "trend_efficiency", "trend_score", "momentum_bull", "momentum_bear", "momentum_strength", "momentum_acceleration", "exhaustion", "directional_efficiency",
        "median_range", "range_std", "recent_long_ratio", "vol_expansion", "vol_contraction", "is_vol_extreme", "is_range", "is_trending_bull", "is_trending_bear", "is_transition",
        "nearest_support_distance", "nearest_resistance_distance", "support_strength", "resistance_strength",
        "doji", "hammer", "shooting_star", "engulfing_bull", "engulfing_bear", "inside", "marubozu",
        "running_present", "running_direction", "running_body_ratio", "running_range", "running_confidence",
    ]
    return {"schema_version": FEATURE_SCHEMA_VERSION, "dtype": "float32", "order": names, "features": [{"name": name, "index": index, "type": "float32", "normalization": "feature-engine normalized; no future statistics", "description": _description(name)} for index, name in enumerate(names)]}


def write_feature_schema(path: str | Path) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(json.dumps(feature_schema(), indent=2) + "\n", encoding="utf-8"); return target


def validate_feature_schema(document: Dict[str, Any], names: Sequence[str]) -> None:
    if document.get("schema_version") != FEATURE_SCHEMA_VERSION: raise ValueError("incompatible feature schema version")
    if list(document.get("order", [])) != list(names): raise ValueError("feature order mismatch")
    if len(document.get("features", [])) != len(names): raise ValueError("feature description count mismatch")
