"""Reproducible metrics and walk-forward evaluation helpers."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import EvaluationReport, SignalDirection


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / max(1, len(labels))


def log_loss(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    total = 0.0
    for p, y in zip(probabilities, labels):
        p = max(1e-12, min(1 - 1e-12, p)); total -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return total / max(1, len(labels))


def ece(probabilities: Sequence[float], labels: Sequence[int], bins: int = 10) -> float:
    total = 0.0; n = len(labels)
    if not n: return 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        ids = [j for j, p in enumerate(probabilities) if lo <= p < hi or (i == bins - 1 and p == hi)]
        if ids:
            total += len(ids) / n * abs(sum(probabilities[j] for j in ids) / len(ids) - sum(labels[j] for j in ids) / len(ids))
    return total


def _bucket(p: float) -> str:
    lo = min(0.9, max(0.5, math.floor(max(p, 0.5) * 20) / 20))
    return f"{int(lo*100)}–{int(min(1.0, lo+.05)*100)}%" if lo < .9 else "90%+"


def evaluate(predictions: Sequence[Optional[int]], probabilities: Sequence[Optional[float]], labels: Sequence[int], regimes: Optional[Sequence[str]] = None) -> EvaluationReport:
    accepted = [(p, q, y, regimes[i] if regimes else "UNKNOWN") for i, (p, q, y) in enumerate(zip(predictions, probabilities, labels)) if p is not None and q is not None]
    n = len(labels); correct = sum(int(p == y) for p, _, y, _ in accepted); incorrect = len(accepted) - correct
    probs = [q for _, q, _, _ in accepted]; ys = [y for _, _, y, _ in accepted]
    conf_buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "accuracy": 0, "calibration_error": 0})
    for p, q, y, _ in accepted:
        conf = max(q, 1 - q); b = _bucket(conf); row = conf_buckets[b]; row["count"] += 1; row["accuracy"] += float(p == y); row["calibration_error"] += abs(conf - float(p == y))
    for row in conf_buckets.values():
        if row["count"]: row["accuracy"] /= row["count"]; row["calibration_error"] /= row["count"]
    by_regime: Dict[str, Dict[str, float]] = defaultdict(lambda: {"count": 0, "accuracy": 0})
    for p, _, y, r in accepted: by_regime[r]["count"] += 1; by_regime[r]["accuracy"] += float(p == y)
    for row in by_regime.values(): row["accuracy"] /= max(1, row["count"])
    positives=[(p,y) for p,_,y,_ in accepted if y==1]; negatives=[(p,y) for p,_,y,_ in accepted if y==0]
    sensitivity=sum(int(p==y) for p,y in positives)/len(positives) if positives else 0.0
    specificity=sum(int(p==y) for p,y in negatives)/len(negatives) if negatives else 0.0
    balanced=(sensitivity+specificity)/2 if positives and negatives else None
    streak = loss = cur = cur_loss = 0
    for p, _, y, _ in accepted:
        if p == y: cur += 1; cur_loss = 0; streak = max(streak, cur)
        else: cur_loss += 1; cur = 0; loss = max(loss, cur_loss)
    return EvaluationReport(n, len(accepted), n - len(accepted), len(accepted) / max(1, n), correct, incorrect, correct / len(accepted) if accepted else None, balanced, brier_score(probs, ys) if accepted else None, log_loss(probs, ys) if accepted else None, ece(probs, ys) if accepted else None, streak, loss, dict(conf_buckets), dict(by_regime), [])


def walk_forward_splits(n: int, train_size: int, validation_size: int, step: Optional[int] = None, purge: int = 1):
    step = step or validation_size; start = train_size
    while start < n:
        train_end = max(0, start - purge); valid_end = min(n, start + validation_size)
        yield range(0, train_end), range(start, valid_end)
        start += step
