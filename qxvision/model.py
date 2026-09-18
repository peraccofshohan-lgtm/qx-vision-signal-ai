"""Small, inspectable model and calibration layer.

The production repository does not ship weights because no real labelled market
set is included. Training creates a candidate artifact only after leakage and
chronological evaluation gates pass. The inference layer therefore reports an
unavailable model rather than inventing a probability.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import FeatureVector, MarketState, ModelPrediction, Regime
from .features import FEATURE_SCHEMA_VERSION

MODEL_VERSION = "v1-untrained"


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(60.0, x)); return 1.0 / (1.0 + z)
    z = math.exp(min(60.0, x)); return z / (1.0 + z)


class LogisticModel:
    def __init__(self, weights: Sequence[float], bias: float, version: str = MODEL_VERSION):
        self.weights = list(weights); self.bias = float(bias); self.version = version

    def probability(self, values: Sequence[float]) -> float:
        if len(values) != len(self.weights): raise ValueError("feature dimension mismatch")
        return _sigmoid(self.bias + sum(w * x for w, x in zip(self.weights, values)))

    def to_json(self) -> dict:
        return {"type": "logistic_regression", "model_version": self.version, "feature_schema_version": FEATURE_SCHEMA_VERSION, "weights": self.weights, "bias": self.bias}

    @classmethod
    def from_json(cls, d: dict) -> "LogisticModel":
        if d.get("feature_schema_version") != FEATURE_SCHEMA_VERSION: raise ValueError("incompatible feature schema")
        return cls(d["weights"], d["bias"], d.get("model_version", "unknown"))


def fit_logistic(samples: Sequence[Sequence[float]], labels: Sequence[int], *, epochs: int = 300, learning_rate: float = 0.08, l2: float = 0.02) -> LogisticModel:
    if not samples or len(samples) != len(labels): raise ValueError("samples and labels must be non-empty and aligned")
    n, d = len(samples), len(samples[0])
    if any(len(row) != d for row in samples): raise ValueError("ragged feature matrix")
    w = [0.0] * d; b = 0.0
    for _ in range(epochs):
        grad = [0.0] * d; gb = 0.0
        for row, y in zip(samples, labels):
            p = _sigmoid(b + sum(a * x for a, x in zip(w, row))); err = p - int(y)
            gb += err
            for j, x in enumerate(row): grad[j] += err * x
        for j in range(d): w[j] -= learning_rate * (grad[j] / n + l2 * w[j])
        b -= learning_rate * gb / n
    return LogisticModel(w, b, "candidate-logistic")


class PlattCalibrator:
    """Validation-only logistic calibration (Platt scaling)."""
    def __init__(self, a: float = 1.0, b: float = 0.0, fitted: bool = False): self.a, self.b, self.fitted = a, b, fitted
    def transform(self, probability: float) -> float:
        if not self.fitted: return max(0.0, min(1.0, probability))
        p = max(1e-6, min(1 - 1e-6, probability))
        return _sigmoid(self.a * math.log(p / (1 - p)) + self.b)
    def fit(self, probabilities: Sequence[float], labels: Sequence[int], epochs: int = 200) -> "PlattCalibrator":
        if len(probabilities) != len(labels) or len(probabilities) < 4: raise ValueError("calibration needs aligned validation samples")
        a, b = 1.0, 0.0
        for _ in range(epochs):
            ga = gb = 0.0
            for p0, y in zip(probabilities, labels):
                p = max(1e-6, min(1 - 1e-6, p0)); x = math.log(p / (1 - p)); q = _sigmoid(a * x + b); e = q - int(y)
                ga += e * x; gb += e
            a -= 0.02 * ga / len(labels); b -= 0.02 * gb / len(labels)
        self.a, self.b, self.fitted = a, b, True
        return self
    def to_json(self) -> dict: return {"method": "platt", "a": self.a, "b": self.b, "fitted": self.fitted}


class FeatureOOD:
    """Robust training-distribution bounds, fit on training data only."""
    def __init__(self, medians: Sequence[float], scales: Sequence[float], threshold: float = 6.0): self.medians, self.scales, self.threshold = list(medians), list(scales), threshold
    @classmethod
    def fit(cls, samples: Sequence[Sequence[float]]) -> "FeatureOOD":
        if not samples: raise ValueError("OOD fit needs samples")
        d = len(samples[0]); med = []; scale = []
        for j in range(d):
            col = [row[j] for row in samples]; m = statistics.median(col); mad = statistics.median([abs(x - m) for x in col])
            med.append(m); scale.append(max(1e-4, 1.4826 * mad))
        return cls(med, scale)
    def score(self, values: Sequence[float]) -> float:
        if len(values) != len(self.medians): return float("inf")
        z = [abs(x - m) / s for x, m, s in zip(values, self.medians, self.scales)]
        return math.sqrt(sum(x * x for x in z) / max(1, len(z)))
    def is_ood(self, values: Sequence[float]) -> bool: return self.score(values) > self.threshold
    def to_json(self) -> dict: return {"medians": self.medians, "scales": self.scales, "threshold": self.threshold}


class Ensemble:
    def __init__(self, models: Sequence[LogisticModel], weights: Optional[Sequence[float]] = None, calibrator: Optional[PlattCalibrator] = None, ood: Optional[FeatureOOD] = None, version: str = MODEL_VERSION):
        self.models = list(models); self.weights = list(weights or [1.0] * len(self.models)); self.calibrator = calibrator; self.ood = ood; self.version = version
        if len(self.weights) != len(self.models): raise ValueError("ensemble weights mismatch")
    def predict(self, vector: FeatureVector, state: MarketState) -> Tuple[List[ModelPrediction], Optional[float], Optional[float], Optional[str]]:
        if not self.models: return [], None, None, "MODEL_ARTIFACT_UNAVAILABLE"
        if vector.schema_version != FEATURE_SCHEMA_VERSION: return [], None, None, "FEATURE_SCHEMA_MISMATCH"
        if self.ood and self.ood.is_ood(vector.values): return [], None, self.ood.score(vector.values), "OUT_OF_DISTRIBUTION"
        preds = [ModelPrediction(f"logistic_{i}", m.probability(vector.values), True) for i, m in enumerate(self.models)]
        total = sum(self.weights) or 1.0; raw = sum(p.probability_up * w for p, w in zip(preds, self.weights)) / total
        calibrated = self.calibrator.transform(raw) if self.calibrator else raw
        variance = sum((p.probability_up - raw) ** 2 for p in preds) / max(1, len(preds))
        agreement = max(0.0, 1.0 - math.sqrt(variance) * 3.0)
        return preds, calibrated, agreement, None


def load_ensemble(directory: str | Path) -> Optional[Ensemble]:
    p = Path(directory)
    meta = p / "metadata.json"; model_path = p / "logistic.json"
    if not meta.exists() or not model_path.exists(): return None
    try:
        metadata = json.loads(meta.read_text()); model = LogisticModel.from_json(json.loads(model_path.read_text()))
        calibrator = None
        cp = p / "calibration.json"
        if cp.exists():
            c = json.loads(cp.read_text()); calibrator = PlattCalibrator(c.get("a", 1), c.get("b", 0), c.get("fitted", False))
        ood = None
        op = p / "ood.json"
        if op.exists():
            o = json.loads(op.read_text()); ood = FeatureOOD(o["medians"], o["scales"], o.get("threshold", 6.0))
        if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION: return None
        return Ensemble([model], [1.0], calibrator, ood, metadata.get("model_version", "unknown"))
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        return None


def save_ensemble(directory: str | Path, ensemble: Ensemble) -> None:
    p = Path(directory); p.mkdir(parents=True, exist_ok=True)
    p.joinpath("metadata.json").write_text(json.dumps({"model_version": ensemble.version, "feature_schema_version": FEATURE_SCHEMA_VERSION, "type": "logistic_regression", "artifact_status": "candidate"}, indent=2))
    if ensemble.models: p.joinpath("logistic.json").write_text(json.dumps(ensemble.models[0].to_json(), indent=2))
    if ensemble.calibrator: p.joinpath("calibration.json").write_text(json.dumps(ensemble.calibrator.to_json(), indent=2))
    if ensemble.ood: p.joinpath("ood.json").write_text(json.dumps(ensemble.ood.to_json(), indent=2))
