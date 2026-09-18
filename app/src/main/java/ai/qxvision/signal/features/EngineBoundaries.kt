package ai.qxvision.signal.features

import ai.qxvision.signal.domain.*

/** Named boundaries make the feature stack replaceable without changing signal policy. */
class MarketStructureEngine(private val featureEngine:FeatureEngine=FeatureEngine()){fun detect(candles:List<ReconstructedCandle>)=featureEngine.state(candles).structure}
class SupportResistanceEngine(private val featureEngine:FeatureEngine=FeatureEngine()){fun detect(candles:List<ReconstructedCandle>)=featureEngine.state(candles).zones}
class MomentumEngine(private val featureEngine:FeatureEngine=FeatureEngine()){fun analyze(candles:List<ReconstructedCandle>)=featureEngine.state(candles).momentum}
class VolatilityEngine(private val featureEngine:FeatureEngine=FeatureEngine()){fun analyze(candles:List<ReconstructedCandle>)=featureEngine.state(candles).volatility}
class TrendEngine(private val featureEngine:FeatureEngine=FeatureEngine()){fun analyze(candles:List<ReconstructedCandle>)=featureEngine.state(candles).trend}
class RegimeDetectionEngine(private val featureEngine:FeatureEngine=FeatureEngine()){fun classify(candles:List<ReconstructedCandle>)=featureEngine.state(candles).regime}
