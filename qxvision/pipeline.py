"""End-to-end offline screenshot analysis with measured stage timings."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set

from .domain import *
from .features import build_feature_vector, build_state
from .image_io import load_image
from .model import Ensemble, load_ensemble, MODEL_VERSION
from .signal import SignalPolicy, decide_with_features
from .vision import analyze as vision_analyze, image_hash


@dataclass
class AnalysisResult:
    decision: SignalDecision
    quality: ScreenshotQualityReport
    sequence: CandleSequence
    state: MarketState
    features: FeatureVector
    stage_ms: Dict[str, float]
    screenshot_hash: str


class VisionSignalPipeline:
    def __init__(self, model_dir: str | Path | None = None, policy: Optional[SignalPolicy] = None):
        self.model_dir = Path(model_dir) if model_dir else None
        self.ensemble: Optional[Ensemble] = load_ensemble(self.model_dir) if self.model_dir else None
        self.policy = policy or SignalPolicy()
        self._seen_hashes: Set[str] = set()

    def analyze_frame(self, frame: ImageFrame, *, asset: Optional[str] = None) -> AnalysisResult:
        timings: Dict[str, float] = {}
        start = time.perf_counter()
        digest = image_hash(frame); duplicate = digest in self._seen_hashes; self._seen_hashes.add(digest)
        t = time.perf_counter(); sequence = vision_analyze(frame, duplicate=duplicate); timings["vision"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter(); state = build_state(sequence.candles); features = build_feature_vector(sequence.candles, sequence.chart.quality, state); timings["features"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        # Only the artifact-backed ensemble produces a probability. No weights
        # are synthesized when the registry is absent.
        decision = decide_with_features(sequence.candles, sequence.chart.quality, state, features, self.ensemble, policy=self.policy, processing_time_ms=0.0, model_version=self.ensemble.version if self.ensemble else MODEL_VERSION)
        timings["decision"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - start) * 1000
        decision.processing_time_ms = timings["total"]
        return AnalysisResult(decision, sequence.chart.quality, sequence, state, features, timings, digest)

    def analyze_path(self, path: str | Path, *, asset: Optional[str] = None) -> AnalysisResult:
        decode_start = time.perf_counter(); frame = load_image(path); decode_ms = (time.perf_counter() - decode_start) * 1000
        result = self.analyze_frame(frame, asset=asset)
        result.stage_ms = {"image_decode": decode_ms, **result.stage_ms}
        result.stage_ms["total"] = decode_ms + result.stage_ms.get("total", 0.0)
        result.decision.processing_time_ms = result.stage_ms["total"]
        return result
