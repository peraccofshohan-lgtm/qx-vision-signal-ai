"""Strict dataset ingestion, versioning, temporal splitting and leakage checks.

No input is silently repaired. Callers may explicitly request sorting after the
validation report has recorded that the source was unsorted.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "asset", "timeframe")
OPTIONAL_COLUMNS = ("volume",)


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

    @property
    def timestamp_dt(self) -> datetime:
        return parse_timestamp(self.timestamp)


@dataclass
class DatasetValidationReport:
    valid: bool
    dataset_id: str
    dataset_version: str
    creation_timestamp: str
    source_description: str
    source_path: str
    source_sha256: str
    assets: List[str]
    timeframes: List[str]
    date_range: Dict[str, Optional[str]]
    row_count: int
    accepted_rows: int
    quarantined_rows: int
    errors: List[str]
    warnings: List[str]
    quarantined: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetValidationError(ValueError):
    def __init__(self, report: DatasetValidationReport):
        self.report = report
        super().__init__("DATASET_VALIDATION_FAILED: " + "; ".join(report.errors))


def parse_timestamp(value: str) -> datetime:
    text = str(value).strip()
    if not text:
        raise ValueError("empty timestamp")
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone is required; naive timestamps are ambiguous")
    return parsed.astimezone(timezone.utc)


def _parse_float(value: Any, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: invalid {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"row {row_number}: {field} is NaN or infinite")
    return result


def _row_from_mapping(mapping: Dict[str, Any], row_number: int) -> OHLCRow:
    missing = [column for column in REQUIRED_COLUMNS if mapping.get(column) in (None, "")]
    if missing:
        raise ValueError(f"row {row_number}: missing values {','.join(missing)}")
    timestamp = str(mapping["timestamp"]).strip()
    parse_timestamp(timestamp)
    asset = str(mapping["asset"]).strip()
    timeframe = str(mapping["timeframe"]).strip()
    if not asset or not timeframe:
        raise ValueError(f"row {row_number}: asset and timeframe must be non-empty")
    row = OHLCRow(timestamp, _parse_float(mapping["open"], "open", row_number), _parse_float(mapping["high"], "high", row_number), _parse_float(mapping["low"], "low", row_number), _parse_float(mapping["close"], "close", row_number), None if mapping.get("volume") in (None, "") else _parse_float(mapping["volume"], "volume", row_number), asset, timeframe)
    if row.high < max(row.open, row.close):
        raise ValueError(f"row {row_number}: high is below open/close")
    if row.low > min(row.open, row.close):
        raise ValueError(f"row {row_number}: low is above open/close")
    if row.high < row.low:
        raise ValueError(f"row {row_number}: high is below low")
    if row.volume is not None and row.volume < 0:
        raise ValueError(f"row {row_number}: volume is negative")
    return row


def _read_csv(path: Path) -> Tuple[List[OHLCRow], List[Dict[str, Any]], List[str]]:
    rows: List[OHLCRow] = []
    quarantined: List[Dict[str, Any]] = []
    errors: List[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fields]
        if missing_columns:
            return [], [], ["MISSING_COLUMNS:" + ",".join(missing_columns)]
        for number, mapping in enumerate(reader, 2):
            try:
                rows.append(_row_from_mapping(mapping, number))
            except ValueError as exc:
                message = str(exc)
                errors.append(message)
                quarantined.append({"row_number": number, "reason": message, "raw": dict(mapping)})
    return rows, quarantined, errors


def _read_parquet(path: Path) -> Tuple[List[OHLCRow], List[Dict[str, Any]], List[str]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore
    except ImportError as exc:
        return [], [], ["PARQUET_REQUIRES_OPTIONAL_PYARROW"]
    table = parquet.read_table(path)
    columns = set(table.column_names)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        return [], [], ["MISSING_COLUMNS:" + ",".join(missing)]
    records = table.to_pylist()
    rows: List[OHLCRow] = []
    quarantined: List[Dict[str, Any]] = []
    errors: List[str] = []
    for number, record in enumerate(records, 1):
        try:
            rows.append(_row_from_mapping(record, number))
        except ValueError as exc:
            errors.append(str(exc)); quarantined.append({"row_number": number, "reason": str(exc), "raw": record})
    return rows, quarantined, errors


def _read_source(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path)
    if suffix in (".parquet", ".pq"):
        return _read_parquet(path)
    return [], [], ["UNSUPPORTED_FORMAT:" + suffix]


def _interval_seconds(timeframe: str) -> Optional[int]:
    match = re.fullmatch(r"\s*(\d+)\s*([smhdw])\s*", timeframe.lower())
    if not match:
        return None
    count, unit = int(match.group(1)), match.group(2)
    return count * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(path: str | Path, *, dataset_version: str = "1.0.0", source_description: str = "user-supplied OHLC dataset", quarantine_invalid: bool = True) -> DatasetValidationReport:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw_rows, quarantined, parse_errors = _read_source(source)
    errors = list(parse_errors); warnings: List[str] = []
    previous_by_asset: Dict[str, datetime] = {}
    seen_keys: set[Tuple[str, str]] = set()
    seen_candles: set[Tuple[str, str, float, float, float, float]] = set()
    assets: set[str] = set(); timeframes: set[str] = set()
    valid_rows: List[OHLCRow] = []
    for row in raw_rows:
        assets.add(row.asset); timeframes.add(row.timeframe)
        timestamp = row.timestamp_dt
        key = (row.asset, row.timestamp)
        candle_key = (row.asset, row.timestamp, row.open, row.high, row.low, row.close)
        if key in seen_keys:
            errors.append(f"DUPLICATE_TIMESTAMP:{row.asset}:{row.timestamp}")
            quarantined.append({"reason": "DUPLICATE_TIMESTAMP", "row": asdict(row)})
            continue
        if candle_key in seen_candles:
            errors.append(f"DUPLICATE_CANDLE:{row.asset}:{row.timestamp}")
            quarantined.append({"reason": "DUPLICATE_CANDLE", "row": asdict(row)})
            continue
        previous = previous_by_asset.get(row.asset)
        if previous is not None and timestamp <= previous:
            errors.append(f"UNSORTED_TIMESTAMP:{row.asset}:{row.timestamp}")
        previous_by_asset[row.asset] = timestamp
        seen_keys.add(key); seen_candles.add(candle_key); valid_rows.append(row)
    if len(assets) > 1:
        warnings.append("MULTIPLE_ASSETS_PRESENT: evaluate assets independently")
    if len(timeframes) > 1:
        errors.append("MIXED_TIMEFRAMES:" + ",".join(sorted(timeframes)))
    for asset in sorted(assets):
        asset_rows = [row for row in valid_rows if row.asset == asset]
        intervals = _interval_seconds(asset_rows[0].timeframe) if asset_rows else None
        if intervals is None and asset_rows:
            warnings.append(f"UNPARSEABLE_TIMEFRAME:{asset}:{asset_rows[0].timeframe}")
        if intervals:
            for previous, current in zip(asset_rows, asset_rows[1:]):
                delta = int((current.timestamp_dt - previous.timestamp_dt).total_seconds())
                if delta <= 0 or delta % intervals != 0:
                    errors.append(f"TIMEFRAME_INCONSISTENCY:{asset}:{previous.timestamp}->{current.timestamp}:{delta}s_expected_multiple_of_{intervals}s")
    sorted_rows = sorted(valid_rows, key=lambda row: (row.asset, row.timestamp_dt))
    timestamps = [row.timestamp_dt.isoformat().replace("+00:00", "Z") for row in sorted_rows]
    dataset_id = _source_sha256(source)
    report = DatasetValidationReport(not errors and not (parse_errors and not valid_rows), dataset_id, dataset_version, datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), source_description, str(source), dataset_id, sorted(assets), sorted(timeframes), {"start": min(timestamps) if timestamps else None, "end": max(timestamps) if timestamps else None}, len(raw_rows), len(valid_rows), len(quarantined), errors, warnings, quarantined)
    if errors and not quarantine_invalid:
        raise DatasetValidationError(report)
    return report


def load_dataset(path: str | Path, *, strict: bool = True, dataset_version: str = "1.0.0", source_description: str = "user-supplied OHLC dataset") -> Tuple[List[OHLCRow], DatasetValidationReport]:
    report = validate_dataset(path, dataset_version=dataset_version, source_description=source_description, quarantine_invalid=True)
    if strict and not report.valid:
        raise DatasetValidationError(report)
    rows, _, _ = _read_source(Path(path))
    # Training never consumes invalid rows in lenient mode either.
    good = {(row.asset, row.timestamp, row.open, row.high, row.low, row.close) for row in rows if all(math.isfinite(value) for value in (row.open, row.high, row.low, row.close)) and row.high >= max(row.open, row.close) and row.low <= min(row.open, row.close) and row.high >= row.low}
    rows = [row for row in rows if (row.asset, row.timestamp, row.open, row.high, row.low, row.close) in good]
    return sorted(rows, key=lambda row: (row.asset, row.timestamp_dt)), report


def load_ohlc_csv(path: str | Path) -> List[OHLCRow]:
    """Backward-compatible strict CSV loader used by the original trainer."""
    rows, report = load_dataset(path, strict=True)
    return rows


def write_dataset_metadata(report: DatasetValidationReport, output: str | Path, *, overwrite: bool = False) -> Path:
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"metadata already exists; refusing silent replacement: {target}")
    target.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return target


def chronological_split(rows: Sequence[OHLCRow], train_fraction: float = .60, validation_fraction: float = .20, purge: int = 1) -> Tuple[List[OHLCRow], List[OHLCRow], List[OHLCRow]]:
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1: raise ValueError("invalid split fractions")
    by_asset: Dict[str, List[OHLCRow]] = {}
    for row in rows: by_asset.setdefault(row.asset, []).append(row)
    train_rows: List[OHLCRow] = []; validation_rows: List[OHLCRow] = []; holdout_rows: List[OHLCRow] = []
    for sequence in by_asset.values():
        ordered=sorted(sequence,key=lambda item:item.timestamp_dt); n=len(ordered); a=int(n*train_fraction); b=int(n*(train_fraction+validation_fraction))
        train_rows.extend(ordered[:a]); validation_rows.extend(ordered[min(n,a+purge):b]); holdout_rows.extend(ordered[min(n,b+purge):])
    return sorted(train_rows,key=lambda item:(item.timestamp_dt,item.asset)), sorted(validation_rows,key=lambda item:(item.timestamp_dt,item.asset)), sorted(holdout_rows,key=lambda item:(item.timestamp_dt,item.asset))


def window_fingerprint(rows: Sequence[OHLCRow]) -> str:
    return hashlib.sha256("|".join(f"{r.timestamp}:{r.open}:{r.high}:{r.low}:{r.close}:{r.asset}" for r in rows).encode()).hexdigest()


def leakage_check(feature_names: Sequence[str], *, target_names: Sequence[str] = ("target", "future", "next_close", "next_open"), feature_timestamps: Optional[Sequence[str]] = None, prediction_timestamps: Optional[Sequence[str]] = None, duplicate_hashes: Optional[Sequence[str]] = None) -> List[str]:
    errors = []
    bad = [name for name in feature_names if any(t in name.lower() for t in target_names)]
    if bad: errors.append("future/target-named feature(s): " + ",".join(bad))
    if feature_timestamps and prediction_timestamps:
        for f, p in zip(feature_timestamps, prediction_timestamps):
            try:
                if parse_timestamp(f) > parse_timestamp(p): errors.append(f"feature timestamp {f} after prediction {p}"); break
            except ValueError as exc:
                errors.append(str(exc)); break
    if duplicate_hashes and len(set(duplicate_hashes)) != len(duplicate_hashes): errors.append("duplicate windows detected")
    return errors


def assert_no_leakage(*args, **kwargs) -> None:
    errors = leakage_check(*args, **kwargs)
    if errors: raise ValueError("LEAKAGE_CHECK_FAILED: " + "; ".join(errors))
