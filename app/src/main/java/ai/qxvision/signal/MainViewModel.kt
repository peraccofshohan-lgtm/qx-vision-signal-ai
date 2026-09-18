package ai.qxvision.signal

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.qxvision.signal.data.PredictionRepository
import ai.qxvision.signal.data.SignalDatabase
import ai.qxvision.signal.domain.*
import ai.qxvision.signal.features.FeatureEngine
import ai.qxvision.signal.ml.LocalModelRunner
import ai.qxvision.signal.ml.EnsembleEngine
import ai.qxvision.signal.signal.SignalEngine
import ai.qxvision.signal.vision.ChartVisionEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface AnalysisStage { data object Idle:AnalysisStage; data object Validating:AnalysisStage; data object Reconstructing:AnalysisStage; data object Extracting:AnalysisStage; data object Ensemble:AnalysisStage; data object FinalDecision:AnalysisStage; data object Complete:AnalysisStage }
data class AnalysisUiState(val stage:AnalysisStage=AnalysisStage.Idle,val bitmap:Bitmap?=null,val result:AnalysisResult?=null,val error:String?=null)

class MainViewModel(app:Application):AndroidViewModel(app){
    private val _state=MutableStateFlow(AnalysisUiState()); val state:StateFlow<AnalysisUiState> = _state.asStateFlow()
    private val vision=ChartVisionEngine(); private val features=FeatureEngine(); private val signal=SignalEngine(); private val model=LocalModelRunner(app); private val ensemble=EnsembleEngine(listOf(model)); private val repo=PredictionRepository(SignalDatabase.get(app).predictions())
    val history=repo.history
    private val prefs=app.getSharedPreferences("qx_settings",0)
    private val _strictness=MutableStateFlow(prefs.getString("strictness","BALANCED")?:"BALANCED")
    val strictness:StateFlow<String> = _strictness.asStateFlow()
    private var previousHash:String?=null
    private var modelWarmed=false
    private fun policy()=when(_strictness.value){"CONSERVATIVE"->SignalEngine.Policy(.70f,.60f,.70f,.34f,10);"CUSTOM"->SignalEngine.Policy(.66f,.55f,.65f,.40f,8);else->SignalEngine.Policy()}
    fun setStrictness(value:String){if(value !in setOf("CONSERVATIVE","BALANCED","CUSTOM"))return; prefs.edit().putString("strictness",value).apply(); _strictness.value=value}

    fun analyze(uri:Uri){ viewModelScope.launch(Dispatchers.Default){
        _state.value=AnalysisUiState(AnalysisStage.Validating)
        try {
            val bitmap=decode(uri) ?: throw IllegalArgumentException("IMAGE_DECODE_FAILED")
            _state.value=AnalysisUiState(AnalysisStage.Validating,bitmap=bitmap)
            val hash=vision.screenshotHash(bitmap); val duplicate=hash==previousHash; previousHash=hash
            _state.value=AnalysisUiState(AnalysisStage.Reconstructing,bitmap=bitmap)
            val start=System.nanoTime(); val visionStart=System.nanoTime(); val sequence=vision.analyze(bitmap,duplicate); val visionEnd=System.nanoTime()
            _state.value=AnalysisUiState(AnalysisStage.Extracting,bitmap=bitmap)
            val featureStart=System.nanoTime(); val market=features.state(sequence.candles); val vector=features.vector(sequence.candles,sequence.chart.quality,market); val featureEnd=System.nanoTime(); if(!modelWarmed){model.warmup(vector.values.size);modelWarmed=true}
            _state.value=AnalysisUiState(AnalysisStage.Ensemble,bitmap=bitmap)
            val decisionStart=System.nanoTime(); _state.value=AnalysisUiState(AnalysisStage.FinalDecision,bitmap=bitmap)
            val decision=signal.decide(sequence,market,vector,ensemble,policy=policy(),elapsedMs=(System.nanoTime()-start)/1_000_000); val decisionEnd=System.nanoTime()
            val result=AnalysisResult(decision,sequence,market,vector,mapOf("vision" to ((visionEnd-visionStart)/1_000_000),"features" to ((featureEnd-featureStart)/1_000_000),"decision" to ((decisionEnd-decisionStart)/1_000_000),"total" to ((decisionEnd-start)/1_000_000)),hash)
            repo.save(result)
            _state.value=AnalysisUiState(AnalysisStage.Complete,bitmap,result)
        }catch(t:Throwable){_state.value=AnalysisUiState(AnalysisStage.Idle,error=t.message?:"ANALYSIS_FAILED")}
    } }

    private fun decode(uri:Uri):Bitmap?{ val cr=getApplication<Application>().contentResolver; val bounds=BitmapFactory.Options().apply{inJustDecodeBounds=true}; cr.openInputStream(uri)?.use{BitmapFactory.decodeStream(it,null,bounds)}; if(bounds.outWidth<=0||bounds.outHeight<=0)return null; var sample=1; while(bounds.outWidth/sample>1600||bounds.outHeight/sample>1600||bounds.outWidth/sample*(bounds.outHeight/sample)>6_000_000)sample*=2; val opts=BitmapFactory.Options().apply{inSampleSize=sample;inPreferredConfig=Bitmap.Config.ARGB_8888}; return cr.openInputStream(uri)?.use{BitmapFactory.decodeStream(it,null,opts)} }
    fun labelOutcome(sampleId:String,outcome:String){ viewModelScope.launch(Dispatchers.IO){ repo.label(sampleId,outcome) } }
    fun reset(){_state.value=AnalysisUiState()}
    override fun onCleared(){model.close();super.onCleared()}
}
