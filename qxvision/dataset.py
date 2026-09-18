"""Dataset ingestion, temporal splitting, deduplication and leakage checks."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "asset", "timeframe")

@dataclass(frozen=True)
class OHLCRow:
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float]
    asset: str
    timeframe: str

    @property
    def direction(self) -> int:
        return 1 if self.close > self.open else 0 if self.close < self.open else -1


def load_ohlc_csv(path: str | Path) -> List[OHLCRow]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing: raise ValueError("missing OHLC columns: " + ",".join(missing))
        rows = []
        for number, r in enumerate(reader, 2):
            try:
                rows.append(OHLCRow(r["timestamp"], float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]), float(r["volume"]) if r.get("volume") else None, r["asset"], r["timeframe"]))
            except (TypeError, ValueError) as e: raise ValueError(f"invalid row {number}: {e}") from e
    rows.sort(key=lambda x: (x.asset, x.timestamp))
    if any(r.high < max(r.open, r.close) or r.low > min(r.open, r.close) for r in rows): raise ValueError("OHLC high/low invariant failed")
    return rows


def chronological_split(rows: Sequence[OHLCRow], train_fraction: float = .60, validation_fraction: float = .20, purge: int = 1) -> Tuple[List[OHLCRow], List[OHLCRow], List[OHLCRow]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1: raise ValueError("invalid split fractions")
    ordered = sorted(rows, key=lambda x: (x.asset, x.timestamp)); n = len(ordered); a = int(n * train_fraction); b = int(n * (train_fraction + validation_fraction))
    return ordered[:a], ordered[min(n, a + purge):b], ordered[min(n, b + purge):]


def window_fingerprint(rows: Sequence[OHLCRow]) -> str:
    return hashlib.sha256("|".join(f"{r.timestamp}:{r.open}:{r.high}:{r.low}:{r.close}:{r.asset}" for r in rows).encode()).hexdigest()


def leakage_check(feature_names: Sequence[str], *, target_names: Sequence[str] = ("target", "future", "next_close", "next_open"), feature_timestamps: Optional[Sequence[str]] = None, prediction_timestamps: Optional[Sequence[str]] = None, duplicate_hashes: Optional[Sequence[str]] = None) -> List[str]:
    errors = []
    bad = [name for name in feature_names if any(t in name.lower() for t in target_names)]
    if bad: errors.append("future/target-named feature(s): " + ",".join(bad))
    if feature_timestamps and prediction_timestamps:
        for f, p in zip(feature_timestamps, prediction_timestamps):
            if f > p: errors.append(f"feature timestamp {f} after prediction {p}"); break
    if duplicate_hashes and len(set(duplicate_hashes)) != len(duplicate_hashes): errors.append("duplicate windows detected")
    return errors


def assert_no_leakage(*args, **kwargs) -> None:
    errors = leakage_check(*args, **kwargs)
    if errors: raise ValueError("LEAKAGE_CHECK_FAILED: " + "; ".join(errors))
