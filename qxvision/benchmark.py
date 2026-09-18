"""Measured repeated benchmark utility; it never reports invented timings."""
from __future__ import annotations
import statistics, time
from typing import Dict, Sequence
from .domain import ImageFrame
from .pipeline import VisionSignalPipeline

def percentile(values, p):
    if not values: return None
    values=sorted(values); index=(len(values)-1)*p; low=int(index); high=min(len(values)-1,low+1); return values[low]+(values[high]-values[low])*(index-low)

def benchmark(frame:ImageFrame, repeats:int=10, pipeline:VisionSignalPipeline|None=None)->Dict[str,Dict[str,float|None]]:
    if repeats<2: raise ValueError("benchmark requires at least two repetitions")
    p=pipeline or VisionSignalPipeline(); samples=[]
    for _ in range(repeats):
        r=p.analyze_frame(frame); samples.append(r.stage_ms)
    keys=sorted({key for row in samples for key in row}); return {key:{"min":min(row.get(key,0) for row in samples),"median":statistics.median(row.get(key,0) for row in samples),"p90":percentile([row.get(key,0) for row in samples],.90),"p95":percentile([row.get(key,0) for row in samples],.95),"max":max(row.get(key,0) for row in samples)} for key in keys}
