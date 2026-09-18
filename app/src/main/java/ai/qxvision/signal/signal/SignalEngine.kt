package ai.qxvision.signal.signal

import ai.qxvision.signal.domain.*
import ai.qxvision.signal.features.FEATURE_SCHEMA_VERSION
import ai.qxvision.signal.ml.EnsembleEngine
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.sqrt

/** Final policy. It never turns a technical score into a probability. */
class SignalEngine {
    data class Policy(val minimumProbability: Float=.64f, val minimumQuality: Float=.52f, val minimumAgreement: Float=.62f, val maximumUncertainty: Float=.42f, val minimumCandles: Int=8)

    fun decide(sequence:CandleSequence, state:MarketState, vector:FeatureVector, ensemble:EnsembleEngine?, policy:Policy=Policy(), elapsedMs:Long):SignalDecision {
        val quality=sequence.chart.quality; val reasons=mutableSetOf<String>(); val bull=mutableListOf<String>();val bear=mutableListOf<String>();val neutral=mutableListOf<String>()
        when(state.trend.classification){"BULLISH","STRONG_BULLISH","WEAK_BULLISH"->bull+="trend ${state.trend.classification.lowercase()}";"BEARISH","STRONG_BEARISH","WEAK_BEARISH"->bear+="trend ${state.trend.classification.lowercase()}";else->neutral+="trend is range or transition"}
        when {state.momentum.bullishScore>state.momentum.bearishScore+.12f->bull+="recent body momentum favors up";state.momentum.bearishScore>state.momentum.bullishScore+.12f->bear+="recent body momentum favors down";else->neutral+="momentum is mixed"}
        when(state.structure.bias){"BULLISH"->bull+="higher-high/higher-low structure";"BEARISH"->bear+="lower-high/lower-low structure";else->neutral+="structure is ambiguous"}
        if(!quality.supported) reasons.addAll(quality.failureReasons.ifEmpty{listOf("SCREENSHOT_NOT_SUPPORTED")}); if(quality.qualityScore<policy.minimumQuality) reasons += "SCREENSHOT_QUALITY_INSUFFICIENT"; if(sequence.candles.size<policy.minimumCandles)reasons+="TOO_FEW_RELIABLE_CANDLES"; if(state.volatility.classification==VolatilityClass.EXTREME)reasons+="EXTREME_VOLATILITY"; if(sequence.candles.lastOrNull()?.let{!it.isComplete&&it.visualConfidence<.35f}==true)reasons+="RUNNING_CANDLE_UNSTABLE"
        if(ensemble==null)reasons+="MODEL_ARTIFACT_UNAVAILABLE"; val output=ensemble?.predict(vector); val predictions=output?.predictions.orEmpty(); if(output?.failure!=null)reasons+=output.failure!!
        val probability=if(ensemble?.hasCalibration==true)output?.probabilityUp?.let{ensemble.calibrated(it)} else null; if(ensemble?.hasCalibration!=true)reasons+="CALIBRATION_ARTIFACT_UNAVAILABLE"; if(state.structure.bias=="NEUTRAL")reasons+="AMBIGUOUS_STRUCTURE"
        val agreement=output?.agreement; if(agreement==null||agreement<policy.minimumAgreement)reasons+="LOW_MODEL_AGREEMENT"
        val visionUncertainty=(1f-quality.qualityScore*sequence.reconstructionConfidence).coerceIn(0f,1f); val aleatoric=(1f-max(state.momentum.strength,state.trend.score)*.55f+state.volatility.expansion*.45f).coerceIn(0f,1f); val uncertainty=(.55f*visionUncertainty+.25f*aleatoric+.20f*if(agreement==null)1f else 0f).coerceIn(0f,1f); if(uncertainty>policy.maximumUncertainty)reasons+="UNCERTAINTY_TOO_HIGH"; if(probability==null)reasons+="CALIBRATED_PROBABILITY_UNAVAILABLE"; else if(max(probability,1-probability)<policy.minimumProbability)reasons+="PROBABILITY_BELOW_POLICY_THRESHOLD"
        val accepted=reasons.isEmpty()&&probability!=null; val direction=if(!accepted)SignalDirection.NO_TRADE else if(probability>=.5f)SignalDirection.UP else SignalDirection.DOWN
        return SignalDecision(direction,probability,probability?.let{max(it,1-it)},!accepted,reasons.toList().sorted(),state.regime,bull,bear,neutral,agreement,uncertainty,quality.qualityScore,elapsedMs,ensemble?.modelVersion?:"v1-untrained",vector.schemaVersion,predictions)
    }
}
