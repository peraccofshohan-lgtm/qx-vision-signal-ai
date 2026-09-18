#!/usr/bin/env python3
"""Import user-labelled screenshot/outcome pairs into a candidate JSONL dataset."""
from __future__ import annotations
import argparse
from qxvision.screenshot_dataset import import_manifest

def main(argv=None):
    parser=argparse.ArgumentParser(); parser.add_argument("manifest",help="CSV: screenshot_path,prediction_timestamp,actual_outcome,asset(optional)"); parser.add_argument("output"); parser.add_argument("--model-dir",default=None); args=parser.parse_args(argv)
    count=import_manifest(args.manifest,args.output,model_dir=args.model_dir); print(f"imported {count} immutable candidate screenshot samples into {args.output}"); return 0
if __name__=='__main__': raise SystemExit(main())
