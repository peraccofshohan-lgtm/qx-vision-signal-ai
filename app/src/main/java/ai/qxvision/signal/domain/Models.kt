package ai.qxvision.signal.domain

import android.graphics.Bitmap

enum class Direction { UP, DOWN, UNKNOWN }
enum class SignalDirection { UP, DOWN, NO_TRADE }
enum class Regime { TRENDING_BULL, TRENDING_BEAR, RANGING, BREAKOUT, POST_BREAKOUT, PULLBACK, REVERSAL_ATTEMPT, VOLATILITY_EXPANSION, VOLATILITY_COMPRESSION, CHOPPY, TRANSITION, UNKNOWN }
enum class VolatilityClass { VERY_LOW, LOW, NORMAL, HIGH, EXTREME, UNKNOWN }

data class ChartScreenshot(val bitmap: Bitmap, val sourceUri: String? = null)
data class ScreenshotQuality(
    val qualityScore: Float,
    val blurScore: Float,
    val chartCoverage: Float,
    val candleVisibility: Float,
    val axisVisibility: Float,
    val indicatorVisibility: Float,
    val occlusionScore: Float,
    val resolutionAdequacy: Float,
    val supported: Boolean,
    val failureReasons: List<String>
)
data class RectI(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val width get() = (right - left).coerceAtLeast(0)
    val height get() = (bottom - top).coerceAtLeast(0)
}
data class DetectedChart(
    val bounds: RectI,
    val backgroundColor: Int,
    val bullishColor: Int?,
    val bearishColor: Int?,
    val gridColor: Int?,
    val quality: ScreenshotQuality
)
data class ReconstructedCandle(
    val index: Int,
    val xCenter: Float,
    val bodyTop: Float,
    val bodyBottom: Float,
    val wickTop: Float,
    val wickBottom: Float,
    val direction: Direction,
    val bodyHeight: Float,
    val upperWickLength: Float,
    val lowerWickLength: Float,
    val totalRange: Float,
    val bodyToRangeRatio: Float,
    val relativeOpen: Float,
    val relativeClose: Float,
    val relativeHigh: Float,
    val relativeLow: Float,
    val isComplete: Boolean,
    val visualConfidence: Float,
    val partialObservation: Boolean
)
data class CandleSequence(val candles: List<ReconstructedCandle>, val chart: DetectedChart, val runningCandleIndex: Int?, val reconstructionConfidence: Float)
data class SwingPoint(val index: Int, val value: Float, val kind: String, val major: Boolean, val prominence: Float)
data class MarketStructure(val swings: List<SwingPoint>, val labels: List<String>, val bias: String, val breakOfStructure: Boolean, val changeOfCharacter: Boolean, val score: Float, val explanation: String)
data class SupportResistanceZone(val kind: String, val low: Float, val high: Float, val strength: Float, val touchCount: Int, val distanceFromCurrent: Float)
data class MomentumState(val bullishScore: Float, val bearishScore: Float, val strength: Float, val acceleration: Float, val exhaustionProbability: Float, val efficiency: Float)
data class VolatilityState(val medianRange: Float, val rangeStd: Float, val recentLongRatio: Float, val expansion: Float, val contraction: Float, val classification: VolatilityClass)
data class TrendState(val shortSlope: Float, val visibleSlope: Float, val efficiency: Float, val smoothness: Float, val classification: String, val score: Float)
data class MarketState(val structure: MarketStructure, val zones: List<SupportResistanceZone>, val momentum: MomentumState, val volatility: VolatilityState, val trend: TrendState, val regime: Regime)
data class FeatureVector(val values: FloatArray, val names: List<String>, val schemaVersion: String, val validMask: BooleanArray, val metadata: Map<String, String>)
data class ModelPrediction(val modelName: String, val probabilityUp: Float, val available: Boolean, val failure: String? = null)
data class SignalDecision(
    val direction: SignalDirection,
    val calibratedProbability: Float?,
    val confidence: Float?,
    val abstained: Boolean,
    val abstentionReasons: List<String>,
    val marketRegime: Regime,
    val bullishEvidence: List<String>,
    val bearishEvidence: List<String>,
    val neutralEvidence: List<String>,
    val modelAgreement: Float?,
    val uncertaintyScore: Float?,
    val screenshotQuality: Float,
    val processingTimeMs: Long,
    val modelVersion: String,
    val featureSchemaVersion: String,
    val modelPredictions: List<ModelPrediction>
)
data class AnalysisResult(val decision: SignalDecision, val sequence: CandleSequence, val state: MarketState, val features: FeatureVector, val timingsMs: Map<String, Long>, val screenshotHash: String)
