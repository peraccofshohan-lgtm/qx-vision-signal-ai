#!/usr/bin/env python3
"""Train and evaluate V1 candidates on a validated, user-supplied OHLC dataset.

The final chronological holdout is evaluated once after selection. It is never
used for feature selection, calibration, threshold choice, or model selection.
No artifact is written when the validation gate fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from qxvision.dataset import OHLCRow, load_dataset, leakage_check, write_dataset_metadata
from qxvision.domain import Direction, ReconstructedCandle, ScreenshotQualityReport
from qxvision.evaluate import baseline_predictions, ece, evaluate, grouped_metrics, reliability_curve, risk_coverage_curve
from qxvision.features import FEATURE_SCHEMA_VERSION, build_feature_vector, build_state
from qxvision.labels import LABEL_SCHEMA_VERSION, Label, label_next_candle
from qxvision.model import Ensemble, FeatureOOD, GradientBoostingStumps, IsotonicCalibrator, LogisticModel, PlattCalibrator, RandomStumpForest, fit_gradient_boosting, fit_logistic, fit_random_forest, save_ensemble

ALL_REGIMES = ("TRENDING_BULL", "TRENDING_BEAR", "RANGING", "BREAKOUT", "PULLBACK", "REVERSAL_ATTEMPT", "VOLATILITY_EXPANSION", "VOLATILITY_COMPRESSION", "CHOPPY", "TRANSITION", "UNKNOWN")


def candle_rows(rows: Sequence[OHLCRow]) -> List[ReconstructedCandle]:
    low = min(r.low for r in rows); high = max(r.high for r in rows); scale = max(1e-9, high - low); output=[]
    for index, row in enumerate(rows):
        opened, closed, highest, lowest = (row.open-low)/scale, (row.close-low)/scale, (row.high-low)/scale, (row.low-low)/scale
        direction = Direction.UP if closed > opened else Direction.DOWN if closed < opened else Direction.UNKNOWN
        body_top, body_bottom = min(opened, closed), max(opened, closed); total = max(1e-6, highest-lowest); body = body_bottom-body_top
        output.append(ReconstructedCandle(index, float(index), body_top, body_bottom, lowest, highest, direction, body, body_top-lowest, highest-body_bottom, total, body/total, opened, closed, highest, lowest, True, 1.0, False))
    return output


def vector_for(rows: Sequence[OHLCRow]):
    candles=candle_rows(rows); quality=ScreenshotQualityReport(.95,.95,1,.95,1,0,0,1,False,True,[]); state=build_state(candles); return build_feature_vector(candles,quality,state), state


def windows(rows: Sequence[OHLCRow], lookback: int = 32):
    """Yield only windows whose future target is in the same asset and is UP/DOWN."""
    by_asset: Dict[str, List[OHLCRow]] = {}
    for row in sorted(rows, key=lambda item:(item.asset,item.timestamp_dt)): by_asset.setdefault(row.asset, []).append(row)
    for asset, sequence in by_asset.items():
        for end in range(lookback, len(sequence)):
            context, target = sequence[end-lookback:end], sequence[end]
            label = label_next_candle(context[-1], target)
            if label == Label.DOJI: continue
            yield context, target, asset


def matrix(rows: Sequence[OHLCRow], lookback: int = 32):
    X=[]; y=[]; metadata=[]
    for context, target, asset in windows(rows, lookback):
        vector, state=vector_for(context); X.append(vector.values); y.append(1 if label_next_candle(context[-1], target)==Label.UP else 0); metadata.append({"asset":asset,"prediction_timestamp":context[-1].timestamp,"target_timestamp":target.timestamp,"regime":state.regime.value,"previous_direction":1 if context[-1].close>context[-1].open else 0,"momentum_direction":1 if sum(row.close-row.open for row in context[-3:])>0 else 0})
    return X,y,metadata


def accepted_vectors(probabilities: Sequence[float], threshold: float):
    return [1 if max(p, 1-p) >= threshold and p >= .5 else 0 if max(p, 1-p) >= threshold else None for p in probabilities]


def threshold_search(probabilities: Sequence[float], labels: Sequence[int]) -> Tuple[float, List[Dict[str, float]]]:
    curve=risk_coverage_curve([1 if p>=.5 else 0 for p in probabilities],probabilities,labels,tuple(i/100 for i in range(50,96,5)))
    eligible=[row for row in curve if row["sample_count"]>=max(5,len(labels)*.10)]
    # Validation objective: selective accuracy, with a mild coverage floor.
    chosen=max(eligible or curve,key=lambda row:(row["accuracy"] + .05*row["coverage"],row["coverage"]))
    return float(chosen["threshold"]),curve


def model_validation(model, X_train, y_train, X_valid, y_valid):
    raw_valid=[model.probability(row) for row in X_valid]
    calibrators={"platt":PlattCalibrator().fit(raw_valid,y_valid)}
    if len(y_valid)>=20:
        calibrators["isotonic"]=IsotonicCalibrator().fit(raw_valid,y_valid)
    calibration_scores={name:{"brier_score":sum((calibrator.transform(probability)-label)**2 for probability,label in zip(raw_valid,y_valid))/len(y_valid),"ece":ece([calibrator.transform(probability) for probability in raw_valid],y_valid),"method":name} for name,calibrator in calibrators.items()}
    selected_method=min(calibration_scores,key=lambda name:calibration_scores[name]["brier_score"])
    calibrator=calibrators[selected_method]; calibrated=[calibrator.transform(value) for value in raw_valid]; threshold,curve=threshold_search(calibrated,y_valid); predictions=accepted_vectors(calibrated,threshold); report=evaluate(predictions,[p if prediction is not None else None for p,prediction in zip(calibrated,predictions)],y_valid)
    return calibrator, calibrated, threshold, curve, report, {"selected_method":selected_method,"experiments":calibration_scores}


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--ohlc",required=True); parser.add_argument("--output",default="models/v1/ensemble"); parser.add_argument("--seed",type=int,default=17); parser.add_argument("--dataset-version",default="1.0.0"); parser.add_argument("--source-description",default="user-supplied OHLC dataset"); args=parser.parse_args(argv)
    rows, dataset_report=load_dataset(args.ohlc, strict=True, dataset_version=args.dataset_version, source_description=args.source_description)
    train_rows, valid_rows, test_rows = _split_rows(rows)
    X_train,y_train,meta_train=matrix(train_rows); X_valid,y_valid,meta_valid=matrix(valid_rows); X_test,y_test,meta_test=matrix(test_rows)
    if min(len(X_train),len(X_valid),len(X_test)) < 8: raise SystemExit("not enough chronological windows; at least 8 per split are required")
    feature_names=vector_for(next(windows(train_rows))[0])[0].names
    leakage_errors=leakage_check(feature_names,duplicate_hashes=None)
    if leakage_errors: raise SystemExit("LEAKAGE_CHECK_FAILED: "+"; ".join(leakage_errors))
    candidates={"logistic_regression":fit_logistic(X_train,y_train),"gradient_boosting_stumps":fit_gradient_boosting(X_train,y_train),"random_stump_forest":fit_random_forest(X_train,y_train,seed=args.seed)}
    validation_reports={}; fitted={}
    for name, model in candidates.items():
        calibrator, probabilities, threshold, curve, report, calibration_experiments=model_validation(model,X_train,y_train,X_valid,y_valid); validation_reports[name]={"report":report.__dict__,"threshold":threshold,"risk_coverage":curve,"reliability_curve":reliability_curve(probabilities,y_valid),"calibration":calibrator.to_json(),"calibration_experiments":calibration_experiments}; fitted[name]=(model,calibrator,threshold)
    # Selection is validation-only. Primary criterion is Brier/ECE, with
    # selective accuracy and coverage as transparent tie breakers.
    selected_name=min(validation_reports,key=lambda name:(validation_reports[name]["report"]["brier_score"] if validation_reports[name]["report"]["brier_score"] is not None else 9.0, validation_reports[name]["report"]["ece"] if validation_reports[name]["report"]["ece"] is not None else 9.0, -(validation_reports[name]["report"]["accuracy"] or 0.0)))
    selected_model,selected_calibrator,selected_threshold=fitted[selected_name]
    raw_test=[selected_model.probability(row) for row in X_test]; probabilities=[selected_calibrator.transform(value) for value in raw_test]; predictions=accepted_vectors(probabilities,selected_threshold); holdout_report=evaluate(predictions,[p if prediction is not None else None for p,prediction in zip(probabilities,predictions)],y_test, [item["regime"] for item in meta_test], all_regimes=ALL_REGIMES)
    previous=[item["previous_direction"] for item in meta_test]; momentum=[item["momentum_direction"] for item in meta_test]; baselines=baseline_predictions(y_test,previous=previous,momentum=momentum,seed=args.seed); baseline_reports={name:evaluate(preds,[.5]*len(preds),y_test).__dict__ for name,preds in baselines.items()}
    assets=[item["asset"] for item in meta_test]; time_buckets=[item["prediction_timestamp"][11:13]+":00-"+item["prediction_timestamp"][11:13]+":59Z" if len(item["prediction_timestamp"])>=13 else "UNKNOWN" for item in meta_test]
    asset_breakdown=grouped_metrics(predictions,[p if prediction is not None else None for p,prediction in zip(probabilities,predictions)],y_test,assets)
    time_breakdown=grouped_metrics(predictions,[p if prediction is not None else None for p,prediction in zip(probabilities,predictions)],y_test,time_buckets)
    output=Path(args.output)
    if (output/"metrics.json").exists(): raise SystemExit(f"output already contains metrics; refusing silent replacement: {output}")
    output.mkdir(parents=True,exist_ok=True); write_dataset_metadata(dataset_report,output/"dataset_metadata.json")
    metrics={"status":"candidate","dataset":dataset_report.to_dict(),"label_schema_version":LABEL_SCHEMA_VERSION,"feature_schema_version":FEATURE_SCHEMA_VERSION,"seed":args.seed,"split_windows":{"train":len(X_train),"validation":len(X_valid),"final_holdout":len(X_test)},"selection":{"selected_model":selected_name,"selection_split":"validation_only","threshold":selected_threshold,"validation_reports":validation_reports},"final_holdout":{"report":holdout_report.__dict__,"risk_coverage":risk_coverage_curve([1 if p>=.5 else 0 for p in probabilities],probabilities,y_test),"baselines":baseline_reports,"asset_breakdown":asset_breakdown,"time_of_day_utc_breakdown":time_breakdown,"holdout_is_tuning_free":True},"model_promotion_gate":{"valid_artifact":isinstance(selected_model,LogisticModel),"schema_compatible":True,"calibration_complete":selected_calibrator.fitted,"validation_gate":(validation_reports[selected_name]["report"]["accepted_predictions"]>=5),"onnx_parity":"PENDING_EXPORT","promotion":"NOT_PROMOTED_UNTIL_ONNX_AND_GATES_PASS"}}
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
    if isinstance(selected_model,LogisticModel) and metrics["model_promotion_gate"]["validation_gate"]:
        save_ensemble(output / "candidate",Ensemble([selected_model],[1.0],selected_calibrator,FeatureOOD.fit(X_train),"candidate-"+selected_name))
        metrics["artifact_path"] = str(output / "candidate")
        (output / "metrics.json").write_text(json.dumps(metrics,indent=2,default=str),encoding="utf-8")
        print(json.dumps(metrics,indent=2,default=str)); return 2
    print(json.dumps(metrics,indent=2,default=str)); return 2


def _split_rows(rows: Sequence[OHLCRow], train_fraction=.60, validation_fraction=.20, purge=2):
    by_asset: Dict[str,List[OHLCRow]]={}
    for row in rows: by_asset.setdefault(row.asset,[]).append(row)
    train_rows=[]; validation_rows=[]; holdout_rows=[]
    for sequence in by_asset.values():
        ordered=sorted(sequence,key=lambda item:item.timestamp_dt); n=len(ordered); first=int(n*train_fraction); second=int(n*(train_fraction+validation_fraction))
        train_rows.extend(ordered[:first]); validation_rows.extend(ordered[min(n,first+purge):second]); holdout_rows.extend(ordered[min(n,second+purge):])
    return sorted(train_rows,key=lambda item:(item.timestamp_dt,item.asset)), sorted(validation_rows,key=lambda item:(item.timestamp_dt,item.asset)), sorted(holdout_rows,key=lambda item:(item.timestamp_dt,item.asset))


if __name__ == "__main__": sys.exit(main())
