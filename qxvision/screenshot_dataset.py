"""Immutable candidate dataset for user-labeled screenshot/outcome pairs."""
from __future__ import annotations
import csv, hashlib, json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .domain import Direction, SignalDirection
from .pipeline import VisionSignalPipeline

@dataclass(frozen=True)
class ScreenshotSample:
    sample_id:str
    screenshot_path:str
    screenshot_hash:str
    prediction_timestamp:str
    asset:Optional[str]
    feature_schema_version:str
    model_version:str
    prediction:str
    probability:Optional[float]
    actual_outcome:str
    quality:float
    candle_count:int
    feature_vector:tuple[float,...]
    imported_at:str


def import_manifest(manifest:str|Path, output:str|Path, *, model_dir:str|Path|None=None)->int:
    target=Path(output)
    if target.exists(): raise FileExistsError(f"candidate dataset already exists; refusing replacement: {target}")
    pipeline=VisionSignalPipeline(model_dir); seen=set(); records=[]
    with Path(manifest).open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle); required={"screenshot_path","prediction_timestamp","actual_outcome"}; missing=required-set(reader.fieldnames or [])
        if missing: raise ValueError("missing manifest columns: "+",".join(sorted(missing)))
        for index,row in enumerate(reader,1):
            actual=row["actual_outcome"].upper();
            if actual not in {"UP","DOWN","DOJI","UNKNOWN"}: raise ValueError(f"row {index}: invalid actual_outcome")
            result=pipeline.analyze_path(row["screenshot_path"],asset=row.get("asset"));
            if result.screenshot_hash in seen: raise ValueError(f"row {index}: duplicate screenshot hash")
            seen.add(result.screenshot_hash)
            if not result.quality.is_supported or len(result.sequence.candles)<8: raise ValueError(f"row {index}: screenshot quality/candle count insufficient")
            records.append(asdict(ScreenshotSample(f"screenshot-{index:06d}",row["screenshot_path"],result.screenshot_hash,row["prediction_timestamp"],row.get("asset") or None,result.features.schema_version,result.decision.model_version,result.decision.direction.value,result.decision.calibrated_probability,actual,result.quality.quality_score,len(result.sequence.candles),tuple(result.features.values),datetime.now(timezone.utc).isoformat())))
    target.parent.mkdir(parents=True,exist_ok=True); target.write_text("\n".join(json.dumps(record,sort_keys=True) for record in records)+("\n" if records else ""),encoding="utf-8"); return len(records)
