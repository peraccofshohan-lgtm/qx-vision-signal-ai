"""Walk-forward evaluation entry point; each fold only sees earlier windows."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .train import matrix
from qxvision.dataset import load_ohlc_csv
from qxvision.evaluate import evaluate, walk_forward_splits
from qxvision.model import fit_logistic

def run(X,y,train_size,validation_size,step=None,purge=1):
    folds=[]
    for train_ids,valid_ids in walk_forward_splits(len(X),train_size,validation_size,step,purge):
        tx=[X[i] for i in train_ids]; ty=[y[i] for i in train_ids]; vx=[X[i] for i in valid_ids]; vy=[y[i] for i in valid_ids]
        if len(tx)<4 or not vx: continue
        m=fit_logistic(tx,ty); probs=[m.probability(x) for x in vx]; pred=[int(p>=.5) for p in probs]
        folds.append(evaluate(pred,probs,vy).__dict__)
    return folds

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--ohlc',required=True);p.add_argument('--train-windows',type=int,default=200);p.add_argument('--validation-windows',type=int,default=50);a=p.parse_args(argv);X,y=matrix(load_ohlc_csv(a.ohlc)); print(json.dumps(run(X,y,a.train_windows,a.validation_windows),indent=2));return 0
if __name__=='__main__':main()
