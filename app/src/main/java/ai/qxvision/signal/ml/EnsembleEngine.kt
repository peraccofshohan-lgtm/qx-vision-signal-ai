package ai.qxvision.signal.ml

import ai.qxvision.signal.domain.FeatureVector
import ai.qxvision.signal.domain.ModelPrediction
import kotlin.math.sqrt

/** Validated component ensemble. A one-component registry is explicit, not disguised as many models. */
data class EnsembleOutput(val predictions:List<ModelPrediction>,val probabilityUp:Float?,val agreement:Float?,val failure:String?)
class EnsembleEngine(private val components:List<LocalModelRunner>){
    val available:Boolean get()=components.isNotEmpty()&&components.any{it.failure==null}
    val modelVersion:String get()=components.joinToString("+"){it.modelVersion}
    val hasCalibration:Boolean get()=components.isNotEmpty()&&components.all{it.hasCalibration}
    fun calibrated(raw:Float)=components.firstOrNull()?.calibrated(raw)?:raw
    fun predict(vector:FeatureVector):EnsembleOutput{
        val predictions=components.map{it.predict(vector)}; val available=predictions.filter{it.available}
        if(available.isEmpty())return EnsembleOutput(predictions,null,null,predictions.firstOrNull()?.failure?:"MODEL_ARTIFACT_UNAVAILABLE")
        val mean=available.map{it.probabilityUp}.average().toFloat(); val variance=available.map{(it.probabilityUp-mean)*(it.probabilityUp-mean)}.average().toFloat(); return EnsembleOutput(predictions,mean,(1f-sqrt(variance)*3f).coerceIn(0f,1f),null)
    }
}
