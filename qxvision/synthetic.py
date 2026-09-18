"""Known-ground-truth OHLC to screenshot-like renderer for CV tests only.

Synthetic images are never counted as real market validation. They test whether
geometric reconstruction survives themes, spacing and compression-like noise.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .domain import ImageFrame
from .image_io import save_ppm

@dataclass(frozen=True)
class OHLC:
    open: float; high: float; low: float; close: float


def render(candles: Sequence[OHLC], *, width: int = 640, height: int = 360, bullish: Tuple[int,int,int] = (35, 205, 125), bearish: Tuple[int,int,int] = (235, 80, 95), background: Tuple[int,int,int] = (20, 24, 32), grid: bool = True, spacing: int = 4, seed: int = 7) -> ImageFrame:
    rng = random.Random(seed)
    pixels = [[background for _ in range(width)] for _ in range(height)]
    left, right, top, bottom = 28, width - 20, 16, height - 20
    if grid:
        gc = tuple(min(255, v + 20) for v in background)
        for y in range(top, bottom, max(20, (bottom-top)//8)):
            for x in range(left, right): pixels[y][x] = gc
        for x in range(left, right, max(25, (right-left)//10)):
            for y in range(top, bottom): pixels[y][x] = gc
    lo = min(c.low for c in candles); hi = max(c.high for c in candles); scale = max(1e-6, hi - lo)
    slot = max(4, (right-left) // max(1, len(candles)))
    body_w = max(2, slot - spacing)
    for i, c in enumerate(candles):
        cx = left + i * slot + slot // 2
        def yy(v): return int(round(bottom - (v-lo)/scale * (bottom-top-1)))
        y_open, y_close, y_high, y_low = yy(c.open), yy(c.close), yy(c.high), yy(c.low)
        color = bullish if c.close >= c.open else bearish
        for y in range(min(y_high,y_low), max(y_high,y_low)+1):
            if 0 <= y < height and 0 <= cx < width: pixels[y][cx] = color
        bt, bb = min(y_open,y_close), max(y_open,y_close)
        for x in range(max(0, cx-body_w//2), min(width, cx+body_w//2+1)):
            for y in range(bt, max(bt+1, bb+1)):
                if 0 <= y < height: pixels[y][x] = color
        # deterministic subtle anti-alias-like edge pixels, without changing
        # the known center geometry.
        for x in (cx-body_w//2-1, cx+body_w//2+1):
            if 0 <= x < width and rng.random() < .35:
                for y in range(bt, max(bt+1, bb+1)):
                    if 0 <= y < height: pixels[y][x] = tuple((a+b)//2 for a,b in zip(color, background))
    return ImageFrame(width, height, tuple(tuple(row) for row in pixels), "synthetic")


def write(candles: Sequence[OHLC], path: str, **kwargs) -> ImageFrame:
    frame = render(candles, **kwargs); save_ppm(frame, path); return frame
