"""Purged walk-forward evaluation with one result object per fold."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .train import matrix
from qxvision.dataset import load_dataset
from qxvision.evaluate import evaluate
from qxvision.model import PlattCalibrator, fit_logistic


def run(X,y,metadata,train_size,validation_size,step=None,purge=1):
    step=step or validation_size; folds=[]; start=train_size; fold_id=0
    while start<len(X):
        train_end=max(0,start-purge); valid_end=min(len(X),start+validation_size)
        train_ids=list(range(train_end)); valid_ids=list(range(start,valid_end))
        if len(train_ids)<8 or not valid_ids: break
        calibration_count=max(4,min(len(train_ids)//5,20)); core_ids=train_ids[:-calibration_count]; calibration_ids=train_ids[-calibration_count:]
        model=fit_logistic([X[i] for i in core_ids],[y[i] for i in core_ids]); calibrator=PlattCalibrator().fit([model.probability(X[i]) for i in calibration_ids],[y[i] for i in calibration_ids]); probabilities=[calibrator.transform(model.probability(X[i])) for i in valid_ids]; predictions=[1 if p>=.5 else 0 for p in probabilities]; report=evaluate(predictions,probabilities,[y[i] for i in valid_ids],[metadata[i]["regime"] for i in valid_ids])
        folds.append({"fold_id":fold_id,"train_start":metadata[train_ids[0]]["prediction_timestamp"],"train_end":metadata[train_ids[-1]]["prediction_timestamp"],"purge_start":metadata[train_end]["prediction_timestamp"] if train_end<len(metadata) else None,"purge_end":metadata[start-1]["prediction_timestamp"],"validation_start":metadata[valid_ids[0]]["prediction_timestamp"],"validation_end":metadata[valid_ids[-1]]["prediction_timestamp"],"sample_count":report.total_samples,"accepted_count":report.accepted_predictions,"accuracy":report.accuracy,"coverage":report.coverage,"brier_score":report.brier_score,"log_loss":report.log_loss,"ece":report.ece})
        fold_id+=1; start+=step
    return folds


def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("--ohlc",required=True); parser.add_argument("--output",default="walk_forward.json"); parser.add_argument("--train-windows",type=int,default=200); parser.add_argument("--validation-windows",type=int,default=50); args=parser.parse_args(argv)
    rows,_=load_dataset(args.ohlc); X,y,metadata=matrix(rows); folds=run(X,y,metadata,args.train_windows,args.validation_windows); Path(args.output).write_text(json.dumps({"method":"purged_walk_forward","purge_gap":1,"folds":folds},indent=2)); print(json.dumps(folds,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
