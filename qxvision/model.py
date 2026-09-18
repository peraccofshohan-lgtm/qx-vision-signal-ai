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
from dataclasses import asdict, dataclass
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


class IsotonicCalibrator:
    """Pool-adjacent-violators calibration, used only with adequate validation data."""
    def __init__(self, x: Sequence[float] = (), y: Sequence[float] = (), fitted: bool = False): self.x, self.y, self.fitted = list(x), list(y), fitted
    def fit(self, probabilities: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        if len(probabilities) != len(labels) or len(probabilities) < 20: raise ValueError("isotonic calibration needs at least 20 aligned validation samples")
        pairs=sorted((float(p),float(y)) for p,y in zip(probabilities,labels)); blocks=[]
        for probability,label in pairs:
            blocks.append([probability,probability,label,1.0])
            while len(blocks)>=2 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
                right=blocks.pop(); left=blocks.pop(); blocks.append([left[0],right[1],left[2]+right[2],left[3]+right[3]])
        self.x=[block[1] for block in blocks]; self.y=[block[2]/block[3] for block in blocks]; self.fitted=True; return self
    def transform(self, probability: float) -> float:
        if not self.fitted or not self.x: return max(0.0,min(1.0,probability))
        if probability <= self.x[0]: return self.y[0]
        for index in range(1,len(self.x)):
            if probability <= self.x[index]:
                width=max(1e-9,self.x[index]-self.x[index-1]); fraction=(probability-self.x[index-1])/width; return self.y[index-1]+fraction*(self.y[index]-self.y[index-1])
        return self.y[-1]
    def to_json(self)->dict: return {"method":"isotonic","x":self.x,"y":self.y,"fitted":self.fitted}


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
            c = json.loads(cp.read_text())
            calibrator = IsotonicCalibrator(c.get("x", []), c.get("y", []), c.get("fitted", False)) if c.get("method") == "isotonic" else PlattCalibrator(c.get("a", 1), c.get("b", 0), c.get("fitted", False))
        ood = None
        op = p / "ood.json"
        if op.exists():
            o = json.loads(op.read_text()); ood = FeatureOOD(o["medians"], o["scales"], o.get("threshold", 6.0))
        if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION: return None
        if int(metadata.get("feature_count", len(model.weights))) != len(model.weights): return None
        schema_path = p / "feature_schema.json"
        if not schema_path.exists(): return None
        from .schema import feature_schema
        schema_document = json.loads(schema_path.read_text())
        if schema_document.get("schema_version") != FEATURE_SCHEMA_VERSION or schema_document.get("order") != feature_schema().get("order"): return None
        return Ensemble([model], [1.0], calibrator, ood, metadata.get("model_version", "unknown"))
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        return None


def save_ensemble(directory: str | Path, ensemble: Ensemble) -> None:
    p = Path(directory); p.mkdir(parents=True, exist_ok=True)
    feature_count = len(ensemble.models[0].weights) if ensemble.models else 0
    p.joinpath("metadata.json").write_text(json.dumps({"model_version": ensemble.version, "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_count": feature_count, "input_name": "features", "output_name": "probability_up", "output_semantics": "calibrated_or_raw_probability_up", "type": "logistic_regression", "artifact_status": "candidate"}, indent=2))
    if ensemble.models: p.joinpath("logistic.json").write_text(json.dumps(ensemble.models[0].to_json(), indent=2))
    if ensemble.calibrator: p.joinpath("calibration.json").write_text(json.dumps(ensemble.calibrator.to_json(), indent=2))
    if ensemble.ood: p.joinpath("ood.json").write_text(json.dumps(ensemble.ood.to_json(), indent=2))
    from .schema import write_feature_schema
    write_feature_schema(p / "feature_schema.json")

@dataclass
class DecisionStump:
    feature_index: int
    threshold: float
    left_value: float
    right_value: float

    def value(self, row: Sequence[float]) -> float:
        return self.left_value if row[self.feature_index] <= self.threshold else self.right_value


class GradientBoostingStumps:
    """Small dependency-free gradient-boosting equivalent for evaluation."""
    def __init__(self, base_logit: float, stumps: Sequence[DecisionStump], learning_rate: float = .08, version: str = "candidate-gradient-stumps"):
        self.base_logit, self.stumps, self.learning_rate, self.version = base_logit, list(stumps), learning_rate, version

    def probability(self, values: Sequence[float]) -> float:
        score = self.base_logit + self.learning_rate * sum(stump.value(values) for stump in self.stumps)
        return _sigmoid(score)


class RandomStumpForest:
    """Deterministic random-stump forest; used as a transparent RF-equivalent baseline."""
    def __init__(self, stumps: Sequence[DecisionStump], version: str = "candidate-random-stump-forest"):
        self.stumps, self.version = list(stumps), version

    def probability(self, values: Sequence[float]) -> float:
        if not self.stumps: return .5
        return max(0.0, min(1.0, sum(_sigmoid(stump.value(values)) for stump in self.stumps) / len(self.stumps)))


def _candidate_thresholds(values: Sequence[float], maximum: int = 12) -> List[float]:
    unique = sorted(set(float(value) for value in values))
    if len(unique) <= 1: return unique
    if len(unique) <= maximum: return [(a + b) / 2 for a, b in zip(unique, unique[1:])]
    return [(unique[int(i * (len(unique) - 1) / maximum)] + unique[min(len(unique) - 1, int((i + 1) * (len(unique) - 1) / maximum))]) / 2 for i in range(maximum)]


def _fit_regression_stump(samples: Sequence[Sequence[float]], targets: Sequence[float], feature_indices: Sequence[int]) -> DecisionStump:
    best: Optional[Tuple[float, DecisionStump]] = None
    for feature in feature_indices:
        column = [row[feature] for row in samples]
        for threshold in _candidate_thresholds(column):
            left = [target for row, target in zip(samples, targets) if row[feature] <= threshold]
            right = [target for row, target in zip(samples, targets) if row[feature] > threshold]
            if not left or not right: continue
            left_value, right_value = sum(left) / len(left), sum(right) / len(right)
            loss = sum((target - left_value) ** 2 for target in left) + sum((target - right_value) ** 2 for target in right)
            candidate = DecisionStump(feature, threshold, left_value, right_value)
            if best is None or loss < best[0]: best = (loss, candidate)
    if best is None:
        mean = sum(targets) / max(1, len(targets)); return DecisionStump(0, float("inf"), mean, mean)
    return best[1]


def fit_gradient_boosting(samples: Sequence[Sequence[float]], labels: Sequence[int], *, rounds: int = 40, learning_rate: float = .08) -> GradientBoostingStumps:
    if not samples: raise ValueError("samples must be non-empty")
    prior = max(1e-5, min(1 - 1e-5, sum(labels) / len(labels))); base = math.log(prior / (1 - prior)); stumps: List[DecisionStump] = []
    scores = [base] * len(labels)
    for _ in range(rounds):
        residuals = [label - _sigmoid(score) for label, score in zip(labels, scores)]
        stump = _fit_regression_stump(samples, residuals, range(len(samples[0]))); stumps.append(stump)
        scores = [score + learning_rate * stump.value(row) for score, row in zip(scores, samples)]
    return GradientBoostingStumps(base, stumps, learning_rate)


def fit_random_forest(samples: Sequence[Sequence[float]], labels: Sequence[int], *, trees: int = 40, seed: int = 17) -> RandomStumpForest:
    if not samples: raise ValueError("samples must be non-empty")
    rng = __import__("random").Random(seed); feature_count = len(samples[0]); stumps=[]
    for _ in range(trees):
        feature = rng.randrange(feature_count); threshold_candidates = _candidate_thresholds([row[feature] for row in samples]); threshold = rng.choice(threshold_candidates) if threshold_candidates else 0.0
        left=[label for row,label in zip(samples,labels) if row[feature]<=threshold]; right=[label for row,label in zip(samples,labels) if row[feature]>threshold]
        stumps.append(DecisionStump(feature,threshold,sum(left)/len(left) if left else .5,sum(right)/len(right) if right else .5))
    return RandomStumpForest(stumps)
