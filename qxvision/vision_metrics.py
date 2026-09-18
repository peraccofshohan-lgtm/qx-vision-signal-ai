"""Metrics for synthetic vision reconstruction, kept separate from market metrics."""
from __future__ import annotations
from typing import Dict, Sequence
from .domain import CandleSequence, Direction
from .synthetic import OHLC


def reconstruction_metrics(expected: Sequence[OHLC], result: CandleSequence) -> Dict[str, float]:
    detected=result.candles; n_expected=len(expected); n_detected=len(detected); matched=min(n_expected,n_detected)
    tp=matched; precision=tp/max(1,n_detected); recall=tp/max(1,n_expected); direction_correct=0; body_error=0.; wick_error=0.
    low=min((row.low for row in expected),default=0.); high=max((row.high for row in expected),default=1.); scale=max(1e-9,high-low)
    for index in range(matched):
        source=expected[index]; actual=detected[index]; expected_direction=Direction.UP if source.close>source.open else Direction.DOWN if source.close<source.open else Direction.UNKNOWN
        direction_correct += int(actual.direction==expected_direction)
        expected_body=abs(source.close-source.open)/scale; expected_wick=((source.high-max(source.open,source.close))+(min(source.open,source.close)-source.low))/scale
        body_error += abs(actual.body_height-expected_body); wick_error += abs((actual.upper_wick_length+actual.lower_wick_length)-expected_wick)
    return {"candle_detection_precision":precision,"candle_detection_recall":recall,"direction_accuracy":direction_correct/max(1,matched),"body_geometry_mae":body_error/max(1,matched),"wick_geometry_mae":wick_error/max(1,matched),"sequence_order_accuracy":1.0 if n_detected==n_expected else matched/max(1,n_expected)}
