#!/usr/bin/env python3
"""Train a candidate on user-supplied OHLC; never uses a future row in features.

Usage:
  python -m training.train --ohlc data/eurusd.csv --output models/v1/ensemble

The command fails closed on leakage, too few samples, or a candidate that does
not beat the previous-chandle baseline on the chronological holdout.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Tuple

from qxvision.dataset import load_ohlc_csv, chronological_split, leakage_check
from qxvision.domain import Direction, ImageFrame, Rect, ReconstructedCandle, ScreenshotQualityReport
from qxvision.features import build_feature_vector, build_state, FEATURE_SCHEMA_VERSION
from qxvision.model import Ensemble, FeatureOOD, PlattCalibrator, fit_logistic, save_ensemble
from qxvision.evaluate import evaluate


def candle_rows(rows):
    lo = min(r.low for r in rows); hi = max(r.high for r in rows); scale = max(1e-9, hi-lo)
    out=[]
    for i,r in enumerate(rows):
        o,c,h,l = (r.open-lo)/scale,(r.close-lo)/scale,(r.high-lo)/scale,(r.low-lo)/scale
        direction = Direction.UP if c > o else Direction.DOWN if c < o else Direction.UNKNOWN
        top,bottom=min(o,c),max(o,c)
        out.append(ReconstructedCandle(i, float(i), top,bottom,l,h,direction,bottom-top,top-l,h-bottom,h-l,bottom-top/max(1e-6,h-l),o,c,h,l,True,1.0,False))
    return out


def vector_for(rows):
    candles=candle_rows(rows); quality=ScreenshotQualityReport(.95,.95,1,.95,1,0,0,1,False,True,[]); state=build_state(candles); return build_feature_vector(candles,quality,state)


def windows(rows, lookback=32):
    # group-by-asset is mandatory so a window cannot cross instruments.
    rows=sorted(rows,key=lambda r:(r.asset,r.timestamp)); by={}
    for r in rows: by.setdefault(r.asset,[]).append(r)
    for asset, seq in by.items():
        for end in range(lookback, len(seq)):
            context=seq[end-lookback:end]; target=seq[end]
            if target.direction < 0: continue
            yield context, target.direction


def matrix(rows):
    X=[]; y=[]
    for context,target in windows(rows):
        v=vector_for(context); X.append(v.values); y.append(target)
    return X,y


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--ohlc",required=True); ap.add_argument("--output",default="models/v1/ensemble"); ap.add_argument("--seed",type=int,default=17); args=ap.parse_args(argv)
    rows=load_ohlc_csv(args.ohlc)
    train,valid,test=chronological_split(rows,purge=2)
    Xtr,ytr=matrix(train); Xv,yv=matrix(valid); Xt,yt=matrix(test)
    if min(len(Xtr),len(Xv),len(Xt)) < 8: raise SystemExit("not enough chronological windows; at least 8 per split are required")
    feature_names=vector_for(next(windows(train))[0]).names
    errors=leakage_check(feature_names,duplicate_hashes=None)
    if errors: raise SystemExit("LEAKAGE_CHECK_FAILED: "+"; ".join(errors))
    model=fit_logistic(Xtr,ytr); valid_raw=[model.probability(x) for x in Xv]; calibrator=PlattCalibrator().fit(valid_raw,yv)
    # OOD bounds are fitted only on train, not on validation or holdout.
    ood=FeatureOOD.fit(Xtr); ensemble=Ensemble([model],[1.0],calibrator,ood,"candidate-logistic")
    probs=[calibrator.transform(model.probability(x)) for x in Xt]; pred=[1 if p>=.5 else 0 for p in probs]
    report=evaluate(pred,probs,yt)
    # The fair baseline is the previous candle direction; reconstruct it from
    # the test windows rather than using any target-derived value.
    prev=[]
    for context,_ in windows(test): prev.append(1 if context[-1].close > context[-1].open else 0)
    baseline=evaluate(prev,[.5]*len(prev),yt)
    metrics={"candidate":report.__dict__,"previous_candle_baseline":baseline.__dict__,"train_windows":len(Xtr),"validation_windows":len(Xv),"holdout_windows":len(Xt),"feature_schema_version":FEATURE_SCHEMA_VERSION,"seed":args.seed,"status":"candidate"}
    if report.accuracy is None or baseline.accuracy is None or report.accuracy <= baseline.accuracy:
        Path(args.output).mkdir(parents=True,exist_ok=True); Path(args.output).joinpath("metrics.json").write_text(json.dumps(metrics,default=lambda o:o,indent=2)); raise SystemExit("candidate did not beat previous-candle baseline; not promoted")
    save_ensemble(args.output,ensemble); Path(args.output).joinpath("metrics.json").write_text(json.dumps(metrics,default=lambda o:o,indent=2)); print(json.dumps(metrics,default=lambda o:o,indent=2)); return 0

if __name__ == "__main__": sys.exit(main())
