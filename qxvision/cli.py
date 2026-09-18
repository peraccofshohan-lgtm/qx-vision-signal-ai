"""Command line entry point for local, reproducible analysis."""
from __future__ import annotations
import argparse, json, sys
from .pipeline import VisionSignalPipeline


def main(argv=None):
    p=argparse.ArgumentParser(prog="qxvision"); sub=p.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze"); a.add_argument("image"); a.add_argument("--model-dir",default=None); a.add_argument("--json",action="store_true")
    args=p.parse_args(argv)
    if args.command == "analyze":
        result=VisionSignalPipeline(args.model_dir).analyze_path(args.image)
        d={"decision":result.decision.direction.value,"calibratedProbability":result.decision.calibrated_probability,"confidence":result.decision.confidence,"abstained":result.decision.abstained,"abstentionReasons":result.decision.abstention_reasons,"regime":result.decision.market_regime.value,"quality":result.decision.screenshot_quality,"modelAgreement":result.decision.model_agreement,"uncertainty":result.decision.uncertainty_score,"candles":len(result.sequence.candles),"runningCandleIndex":result.sequence.running_candle_index,"timingsMs":result.stage_ms,"modelVersion":result.decision.model_version}
        print(json.dumps(d,indent=2) if args.json else json.dumps(d))
        return 0
    return 2

if __name__ == "__main__": sys.exit(main())
