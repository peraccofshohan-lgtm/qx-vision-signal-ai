package ai.qxvision.signal

import ai.qxvision.signal.domain.*
import ai.qxvision.signal.features.FeatureEngine
import ai.qxvision.signal.signal.SignalEngine
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class SignalEngineTest {
    @Test fun missingArtifactIsNoTrade(){
        val candles=(0 until 12).map{ ReconstructedCandle(it,it.toFloat(),.4f,.5f,.3f,.6f,Direction.UP,.1f,.1f,.1f,.3f,.33f,.4f,.5f,.6f,.3f,true,.9f,false) }
        val quality=ScreenshotQuality(0.9f,0.8f,1f,0.9f,0.5f,0f,0f,1f,true,emptyList())
        val chart=DetectedChart(RectI(0,0,100,100),0,0xFF00AA00.toInt(),0xFFAA0000.toInt(),null,quality)
        val seq=CandleSequence(candles,chart,11,.88f); val state=FeatureEngine().state(candles); val vector=FeatureEngine().vector(candles,quality,state)
        val result=SignalEngine().decide(seq,state,vector,null,elapsedMs=4)
        assertEquals(SignalDirection.NO_TRADE,result.direction); assertTrue(result.abstentionReasons.contains("MODEL_ARTIFACT_UNAVAILABLE")); assertEquals(null,result.calibratedProbability)
    }
}
