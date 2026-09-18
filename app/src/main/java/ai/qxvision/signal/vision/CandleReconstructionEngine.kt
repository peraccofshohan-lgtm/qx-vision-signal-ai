package ai.qxvision.signal.vision

import android.graphics.Bitmap
import ai.qxvision.signal.domain.*

/** Stable boundary for a future native/OpenCV implementation. */
class CandleReconstructionEngine(private val vision:ChartVisionEngine=ChartVisionEngine()){
    fun reconstruct(bitmap:Bitmap,chart:DetectedChart):CandleSequence=vision.reconstruct(bitmap,chart)
}
