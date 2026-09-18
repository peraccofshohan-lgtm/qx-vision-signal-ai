"""Versioned deterministic next-candle target generation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Sequence

from .dataset import OHLCRow

LABEL_SCHEMA_VERSION = "next_candle_direction.v1"


class TargetDefinition(str, Enum):
    NEXT_OPEN_CLOSE = "next_open_close"
    NEXT_CLOSE_CURRENT_CLOSE = "next_close_current_close"


class Label(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    DOJI = "DOJI"
    EXCLUDED = "EXCLUDED"


@dataclass(frozen=True)
class LabelledWindow:
    asset: str
    prediction_timestamp: str
    target_timestamp: str
    label: Label
    target_definition: TargetDefinition
    schema_version: str = LABEL_SCHEMA_VERSION


def label_next_candle(current: OHLCRow, following: OHLCRow, definition: TargetDefinition = TargetDefinition.NEXT_OPEN_CLOSE) -> Label:
    if current.asset != following.asset:
        raise ValueError("target crossed asset boundary")
    if definition == TargetDefinition.NEXT_OPEN_CLOSE:
        delta = following.close - following.open
    elif definition == TargetDefinition.NEXT_CLOSE_CURRENT_CLOSE:
        delta = following.close - current.close
    else:
        raise ValueError(f"unsupported target definition: {definition}")
    return Label.UP if delta > 0 else Label.DOWN if delta < 0 else Label.DOJI


def make_labels(rows: Sequence[OHLCRow], definition: TargetDefinition = TargetDefinition.NEXT_OPEN_CLOSE) -> List[LabelledWindow]:
    by_asset: dict[str, list[OHLCRow]] = {}
    for row in sorted(rows, key=lambda item: (item.asset, item.timestamp_dt)):
        by_asset.setdefault(row.asset, []).append(row)
    labels: List[LabelledWindow] = []
    for asset, sequence in by_asset.items():
        for current, following in zip(sequence, sequence[1:]):
            label = label_next_candle(current, following, definition)
            labels.append(LabelledWindow(asset, current.timestamp, following.timestamp, label, definition))
    return labels


def binary_labels(rows: Sequence[OHLCRow], definition: TargetDefinition = TargetDefinition.NEXT_OPEN_CLOSE):
    """Return only UP/DOWN samples; exact dojis are intentionally excluded."""
    return [item for item in make_labels(rows, definition) if item.label in (Label.UP, Label.DOWN)]
