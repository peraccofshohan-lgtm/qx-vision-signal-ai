#!/usr/bin/env python3
"""Promote a candidate only after explicit validation and ONNX parity gates."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

REQUIRED=("model.onnx","metadata.json","feature_schema.json","calibration.json","onnx_parity.json")

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--candidate",required=True); p.add_argument("--champion",required=True); args=p.parse_args(argv)
    candidate=Path(args.candidate); champion=Path(args.champion)
    missing=[name for name in REQUIRED if not (candidate/name).exists()]
    if missing: raise SystemExit("PROMOTION_BLOCKED_MISSING:"+",".join(missing))
    metrics=json.loads((candidate/"metrics.json").read_text()) if (candidate/"metrics.json").exists() else {}
    parity=json.loads((candidate/"onnx_parity.json").read_text())
    gate=metrics.get("model_promotion_gate",{})
    if not parity.get("passed"): raise SystemExit("PROMOTION_BLOCKED_ONNX_PARITY")
    if not gate.get("schema_compatible") or not gate.get("calibration_complete") or not gate.get("validation_gate"): raise SystemExit("PROMOTION_BLOCKED_VALIDATION_GATE")
    if not metrics.get("final_holdout",{}).get("holdout_is_tuning_free"): raise SystemExit("PROMOTION_BLOCKED_HOLDOUT_STATUS")
    champion.mkdir(parents=True,exist_ok=True)
    for name in REQUIRED:
        shutil.copy2(candidate/name,champion/name)
    metadata=json.loads((champion/"metadata.json").read_text()); metadata["artifact_status"]="champion"; metadata["promoted_from"] = str(candidate); (champion/"metadata.json").write_text(json.dumps(metadata,indent=2))
    print(json.dumps({"promoted":True,"champion":str(champion),"model_version":metadata.get("model_version")},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
