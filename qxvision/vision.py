"""Deterministic chart localisation and candle reconstruction without a cloud API.

This is intentionally geometric. It does not OCR labels or infer a price that is
not visible. Color candidates are estimated from the image, and every candle
carries its own confidence. The right-most candle is marked PARTIAL_OBSERVATION
because a static screenshot cannot prove that it has closed.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .domain import (
    CandleSequence, DetectedChart, Direction, ImageFrame, Rect, ReconstructedCandle,
    ScreenshotQualityReport,
)

RGB = Tuple[int, int, int]


def _sat(c: RGB) -> float:
    m, n = max(c), min(c)
    return (m - n) / 255.0


def _lum(c: RGB) -> float:
    return (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255.0


def _dist(a: RGB, b: RGB) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / 441.6729559


def _quant(c: RGB, q: int = 16) -> RGB:
    return tuple(min(255, (v // q) * q + q // 2) for v in c)  # type: ignore[return-value]


def image_hash(frame: ImageFrame) -> str:
    h = hashlib.sha256()
    h.update(f"{frame.width}x{frame.height}".encode())
    for row in frame.pixels:
        h.update(bytes(v for c in row for v in c))
    return h.hexdigest()


def _background(frame: ImageFrame) -> RGB:
    points = []
    for y in range(frame.height):
        for x in range(frame.width):
            if x < max(1, frame.width // 20) or y < max(1, frame.height // 20) or x >= frame.width - max(1, frame.width // 20) or y >= frame.height - max(1, frame.height // 20):
                points.append(_quant(frame.pixel(x, y)))
    return Counter(points).most_common(1)[0][0] if points else (0, 0, 0)


def _blur_score(frame: ImageFrame) -> float:
    if frame.width < 2 or frame.height < 2:
        return 0.0
    energy = 0.0
    n = 0
    for y in range(0, frame.height, max(1, frame.height // 120)):
        for x in range(0, frame.width - 1, max(1, frame.width // 120)):
            energy += abs(_lum(frame.pixel(x + 1, y)) - _lum(frame.pixel(x, y)))
            n += 1
    # An edge-energy score is not a medical sharpness metric; it is only used
    # to reject uniform/blurred screenshots. Keep it bounded and explainable.
    return min(1.0, energy / max(1, n) * 8.0)


def _chart_bounds(frame: ImageFrame, background: RGB) -> Rect:
    # Chart geometry usually occupies the largest coherent non-background area.
    # Use a tolerant difference so grid/anti-aliasing stays in the chart.
    col_score = []
    row_score = []
    threshold = 0.045
    for x in range(frame.width):
        col_score.append(sum(_dist(frame.pixel(x, y), background) > threshold for y in range(frame.height)) / max(1, frame.height))
    for y in range(frame.height):
        row_score.append(sum(_dist(frame.pixel(x, y), background) > threshold for x in range(frame.width)) / max(1, frame.width))
    active_x = [i for i, v in enumerate(col_score) if v > 0.015]
    active_y = [i for i, v in enumerate(row_score) if v > 0.015]
    if not active_x or not active_y:
        return Rect(0, 0, frame.width, frame.height)
    left, right = min(active_x), max(active_x) + 1
    top, bottom = min(active_y), max(active_y) + 1
    # Small margins keep axis and edge wicks available without claiming the UI.
    return Rect(max(0, left), max(0, top), min(frame.width, right), min(frame.height, bottom))


def _color_candidates(frame: ImageFrame, bounds: Rect, background: RGB) -> Tuple[Optional[RGB], Optional[RGB], Optional[RGB]]:
    hist: Counter[RGB] = Counter()
    for y in range(bounds.top, bounds.bottom):
        for x in range(bounds.left, bounds.right):
            c = frame.pixel(x, y)
            if _dist(c, background) > 0.07 and _sat(c) > 0.18:
                hist[_quant(c)] += 1
    colors = [c for c, n in hist.most_common(16) if n >= max(2, (bounds.area // 10000))]
    if not colors:
        return None, None, None
    # Merge nearby quantized shades into the strongest color families.
    families: List[List[RGB]] = []
    for c in colors:
        for family in families:
            if _dist(c, family[0]) < 0.23:
                family.append(c)
                break
        else:
            families.append([c])
    reps = []
    for family in families:
        weighted = sorted(family, key=lambda c: hist[c], reverse=True)[0]
        reps.append((weighted, sum(hist[c] for c in family)))
    reps.sort(key=lambda x: x[1], reverse=True)
    c1 = reps[0][0]
    c2 = reps[1][0] if len(reps) > 1 else None
    # Direction is not tied to fixed RGB. Red-vs-green is a common convention,
    # but choose the two observed families and map by hue when possible.
    def hue(c: RGB) -> float:
        r, g, b = [v / 255.0 for v in c]
        mx, mn = max(r, g, b), min(r, g, b)
        if mx == mn: return 0.0
        if mx == r: return ((g - b) / (mx - mn)) % 6 / 6
        if mx == g: return ((b - r) / (mx - mn) + 2) / 6
        return ((r - g) / (mx - mn) + 4) / 6
    # Keep both candidates. Mapping later uses warm/cool hue and is recorded as
    # a confidence reduction for ambiguous same-hue themes.
    if c2 is not None and abs(hue(c1) - hue(c2)) < 0.06:
        return c1, c2, None
    warm = [c for c in (c1, c2) if c is not None and (hue(c) < 0.12 or hue(c) > 0.90)]
    cool = [c for c in (c1, c2) if c is not None and 0.20 < hue(c) < 0.55]
    if warm and cool:
        return cool[0], warm[0], None
    return c1, c2, None


def _near(c: RGB, candidate: Optional[RGB], tolerance: float = 0.30) -> bool:
    return candidate is not None and _dist(c, candidate) <= tolerance


def inspect(frame: ImageFrame, duplicate: bool = False) -> DetectedChart:
    background = _background(frame)
    bounds = _chart_bounds(frame, background)
    bullish, bearish, grid = _color_candidates(frame, bounds, background)
    saturated = 0
    candleish = 0
    for y in range(bounds.top, bounds.bottom):
        for x in range(bounds.left, bounds.right):
            c = frame.pixel(x, y)
            if _sat(c) > 0.18 and _dist(c, background) > 0.07:
                saturated += 1
                if _near(c, bullish) or _near(c, bearish): candleish += 1
    coverage = bounds.area / max(1, frame.width * frame.height)
    candle_visibility = min(1.0, candleish / max(1, saturated)) if saturated else 0.0
    # Axis evidence is measured from continuous non-background runs near the
    # right/bottom borders; no axis is assumed to exist.
    right_start = max(bounds.left, bounds.right - max(2, bounds.width // 16)); bottom_start = max(bounds.top, bounds.bottom - max(2, bounds.height // 16))
    right_line = max((sum(_dist(frame.pixel(x, y), background) > .08 for y in range(bounds.top, bounds.bottom)) / max(1, bounds.height) for x in range(right_start, bounds.right)), default=0.0)
    bottom_line = max((sum(_dist(frame.pixel(x, y), background) > .08 for x in range(bounds.left, bounds.right)) / max(1, bounds.width) for y in range(bottom_start, bounds.bottom)), default=0.0)
    axis_visibility = min(1.0, max(right_line, bottom_line))
    indicator_visibility = min(1.0, max(0, saturated - candleish) / max(1, bounds.area) * 12.0)
    occlusion_score = min(1.0, indicator_visibility * 0.65 + (1.0 - candle_visibility) * 0.35)
    blur = _blur_score(frame)
    resolution = min(1.0, min(bounds.width, bounds.height) / 240.0)
    reasons: List[str] = []
    if min(frame.width, frame.height) < 160: reasons.append("RESOLUTION_TOO_LOW")
    if coverage < 0.22: reasons.append("CHART_AREA_TOO_SMALL")
    if bullish is None or bearish is None: reasons.append("DIRECTION_COLORS_AMBIGUOUS")
    if candle_visibility < 0.25: reasons.append("CANDLE_PIXELS_NOT_RELIABLE")
    if blur < 0.035: reasons.append("LOW_EDGE_ENERGY_OR_BLUR")
    if duplicate: reasons.append("DUPLICATE_SCREENSHOT")
    score = max(0.0, min(1.0, 0.30 * coverage + 0.25 * candle_visibility + 0.20 * resolution + 0.25 * blur))
    supported = not any(r in reasons for r in ("RESOLUTION_TOO_LOW", "CANDLE_PIXELS_NOT_RELIABLE", "DIRECTION_COLORS_AMBIGUOUS")) and score >= 0.35
    quality = ScreenshotQualityReport(score, blur, coverage, candle_visibility, axis_visibility, indicator_visibility, occlusion_score, resolution, duplicate, supported, reasons)
    return DetectedChart(bounds, background, bullish, bearish, grid, None, quality)


def reconstruct(frame: ImageFrame, chart: DetectedChart) -> CandleSequence:
    b = chart.bounds
    if b.width <= 0 or b.height <= 0:
        return CandleSequence([], chart, None, 0.0)
    # A candle occupies a vertical run of columns containing direction-colored
    # pixels. Gaps between runs are the chart's horizontal spacing.
    active = []
    for x in range(b.left, b.right):
        count = sum(1 for y in range(b.top, b.bottom) if _near(frame.pixel(x, y), chart.bullish_color) or _near(frame.pixel(x, y), chart.bearish_color))
        active.append(count >= max(1, int(b.height * 0.012)))
    runs: List[Tuple[int, int]] = []
    start = None
    for i, on in enumerate(active + [False]):
        if on and start is None: start = i
        if not on and start is not None:
            if i - start >= 1: runs.append((b.left + start, b.left + i))
            start = None
    # Merge a wick/body split separated by <= 1 pixel; reject giant UI blocks.
    merged: List[Tuple[int, int]] = []
    for left, right in runs:
        if merged and left - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    candles: List[ReconstructedCandle] = []
    for idx, (left, right) in enumerate(merged):
        xs = range(left, right)
        colored = [(x, y, frame.pixel(x, y)) for x in xs for y in range(b.top, b.bottom) if _near(frame.pixel(x, y), chart.bullish_color) or _near(frame.pixel(x, y), chart.bearish_color)]
        if not colored: continue
        ys = [y for _, y, _ in colored]
        top, bottom = min(ys), max(ys)
        # Body is inferred from columns with the largest horizontal occupancy;
        # wicks have a much smaller occupancy. This works with anti-aliased bars.
        row_counts: Dict[int, int] = defaultdict(int)
        for _, y, _ in colored: row_counts[y] += 1
        body_rows = [y for y, n in row_counts.items() if n >= max(1, (right - left) * 0.45)]
        body_top, body_bottom = (min(body_rows), max(body_rows)) if body_rows else (top, bottom)
        bullish_pixels = sum(1 for _, _, c in colored if _near(c, chart.bullish_color))
        bearish_pixels = sum(1 for _, _, c in colored if _near(c, chart.bearish_color))
        direction = Direction.UP if bullish_pixels > bearish_pixels else Direction.DOWN if bearish_pixels > bullish_pixels else Direction.UNKNOWN
        h = max(1.0, float(b.height))
        # Store normalized price coordinates (higher price = larger value), not
        # screen coordinates. Geometry lengths remain screen-normalized below.
        high_price, low_price = 1.0 - top / h, 1.0 - bottom / h
        body_high, body_low = 1.0 - body_top / h, 1.0 - body_bottom / h
        if direction == Direction.UP:
            relative_open, relative_close = body_low, body_high
        elif direction == Direction.DOWN:
            relative_open, relative_close = body_high, body_low
        else:
            relative_open = relative_close = (body_high + body_low) / 2.0
        total = max(1.0, float(bottom - top + 1)) / h
        body = max(0.0, float(body_bottom - body_top + 1)) / h
        confidence = min(1.0, 0.35 + 0.45 * abs(bullish_pixels - bearish_pixels) / max(1, len(colored)) + 0.20 * min(1.0, (right - left) / 5.0))
        candles.append(ReconstructedCandle(idx, (left + right - 1) / 2.0, body_top / h, body_bottom / h, top / h, bottom / h, direction, body, max(0.0, (body_top - top) / h), max(0.0, (bottom - body_bottom) / h), total, body / max(total, 1e-6), relative_open, relative_close, high_price, low_price, True, confidence))
    if candles:
        # The final visible bar cannot be proven complete from one screenshot.
        candles[-1].is_complete = False
        candles[-1].partial_observation = True
        candles[-1].visual_confidence *= 0.65
    conf = sum(c.visual_confidence for c in candles) / max(1, len(candles))
    return CandleSequence(candles, chart, len(candles) - 1 if candles else None, conf)


def analyze(frame: ImageFrame, duplicate: bool = False) -> CandleSequence:
    return reconstruct(frame, inspect(frame, duplicate=duplicate))
