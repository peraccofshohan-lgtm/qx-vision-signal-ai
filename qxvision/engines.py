"""Named engine facades keep stable boundaries for future V2 components."""
from __future__ import annotations
from typing import Sequence
from .domain import *
from .features import market_structure, support_resistance, momentum, volatility, trend, regime
from .vision import inspect, reconstruct

class CandleReconstructionEngine:
    def reconstruct(self, frame:ImageFrame, chart:DetectedChart|None=None)->CandleSequence:
        detected=chart or inspect(frame)
        return reconstruct(frame,detected)

class MarketStructureEngine:
    def analyze(self,candles:Sequence[ReconstructedCandle])->MarketStructure:return market_structure(candles)

class SupportResistanceEngine:
    def detect(self,candles:Sequence[ReconstructedCandle])->list[SupportResistanceZone]:return support_resistance(candles)

class MomentumEngine:
    def analyze(self,candles:Sequence[ReconstructedCandle])->MomentumState:return momentum(candles)

class VolatilityEngine:
    def analyze(self,candles:Sequence[ReconstructedCandle])->VolatilityState:return volatility(candles)

class TrendEngine:
    def analyze(self,candles:Sequence[ReconstructedCandle])->TrendState:return trend(candles)

class RegimeDetectionEngine:
    def classify(self,state:MarketState)->Regime:return regime(state)
