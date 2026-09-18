package ai.qxvision.signal

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.qxvision.signal.data.PredictionEntity

private val Ink=Color(0xFF0A0E14); private val Panel=Color(0xFF121923); private val Muted=Color(0xFF8995A7); private val Teal=Color(0xFF55D6BE); private val Red=Color(0xFFFF7186)

class MainActivity:ComponentActivity(){
    private val vm by viewModels<MainViewModel>()
    override fun onCreate(savedInstanceState:Bundle?){super.onCreate(savedInstanceState); val incoming=if(intent.action==Intent.ACTION_SEND)intent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM) else null; setContent{QxTheme{QxApp(vm,incoming)}}}
}

@Composable private fun QxTheme(content: @Composable () -> Unit){MaterialTheme(colorScheme=darkColorScheme(background=Ink,surface=Panel,primary=Teal,onPrimary=Ink,secondary=Teal,onBackground=Color(0xFFEAF0F7),onSurface=Color(0xFFEAF0F7),error=Red),content=content)}

@Composable private fun QxApp(vm:MainViewModel,incoming:Uri?){
    var page by remember{mutableStateOf("HOME")}; val state by vm.state.collectAsState(); val history by vm.history.collectAsState(initial=emptyList()); val strictness by vm.strictness.collectAsState(); val picker=rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()){it?.let{vm.analyze(it);page="ANALYSIS"}}
    LaunchedEffect(incoming){incoming?.let{vm.analyze(it);page="ANALYSIS"}}
    Scaffold(containerColor=Ink){pad->Column(Modifier.padding(pad).fillMaxSize().padding(horizontal=20.dp)){
        Row(Modifier.fillMaxWidth().padding(top=22.dp,bottom=12.dp),verticalAlignment=Alignment.CenterVertically){Column(Modifier.weight(1f)){Text("QX VISION",color=Color.White,fontSize=21.sp,fontWeight=FontWeight.Bold);Text("SIGNAL AI  /  V1.0.0",color=Teal,fontSize=10.sp,letterSpacing=2.sp)}Text("LOCAL",color=Teal,fontSize=11.sp,fontWeight=FontWeight.Bold)}
        Row(horizontalArrangement=Arrangement.spacedBy(8.dp),modifier=Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()).padding(bottom=16.dp)){listOf("HOME","ANALYSIS","HISTORY","PERFORMANCE","MODEL","SETTINGS").forEach{p->TextButton(onClick={page=p}){Text(p,color=if(page==p)Teal else Muted,fontSize=11.sp)}}}
        when(page){"HOME"->Home({picker.launch(ActivityResultContracts.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly));page="ANALYSIS"},{page="HISTORY"});"ANALYSIS"->Analysis(state,{picker.launch(ActivityResultContracts.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))},{page="HOME"});"HISTORY"->History(history,{id,outcome->vm.labelOutcome(id,outcome)});"PERFORMANCE"->Performance(history);"MODEL"->ModelInfo(history.size);"SETTINGS"->Settings(strictness,vm::setStrictness);else->Home({picker.launch(ActivityResultContracts.PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))},{page="HISTORY"})}
    }}
}

@Composable private fun Home(analyze:()->Unit,history:()->Unit){LazyColumn(verticalArrangement=Arrangement.spacedBy(14.dp),modifier=Modifier.fillMaxSize()){
    item{Spacer(Modifier.height(18.dp));Text("SCREENSHOT-BASED",color=Muted,fontSize=12.sp,letterSpacing=2.sp);Text("NEXT-CANDLE ANALYSIS",color=Color.White,fontSize=28.sp,fontWeight=FontWeight.Bold);Text("A local, uncertainty-aware research instrument for the next completed one-minute candle.",color=Muted,fontSize=14.sp,modifier=Modifier.padding(top=8.dp))}
    item{Button(onClick=analyze,modifier=Modifier.fillMaxWidth().height(58.dp),shape=RoundedCornerShape(12.dp),colors=ButtonDefaults.buttonColors(containerColor=Teal,contentColor=Ink)){Text("ANALYZE CHART",fontWeight=FontWeight.Bold,letterSpacing=1.sp)}}
    item{OutlinedButton(onClick=history,modifier=Modifier.fillMaxWidth().height(52.dp),shape=RoundedCornerShape(12.dp)){Text("HISTORY  ·  PERFORMANCE",color=Color.White)}}
    item{InfoCard("MODEL STATUS","UNAVAILABLE / NO TRAINED ARTIFACT", "A validated probability model is not bundled without labelled market data. The app abstains rather than fabricate a signal.")}
    item{Row(horizontalArrangement=Arrangement.spacedBy(10.dp),modifier=Modifier.fillMaxWidth()){Metric("MODE","LOCAL",Modifier.weight(1f));Metric("HORIZON","NEXT 1M",Modifier.weight(1f));Metric("POLICY","ABSTAIN",Modifier.weight(1f))}}
    item{Text("Privacy by default: screenshots stay on device. No credentials, trade controls, cloud API, or automatic execution.",color=Muted,fontSize=12.sp,modifier=Modifier.padding(vertical=14.dp))}
}}

@Composable private fun Analysis(state:AnalysisUiState,pick:()->Unit,home:()->Unit){LazyColumn(verticalArrangement=Arrangement.spacedBy(12.dp),modifier=Modifier.fillMaxSize()){
    item{Text("ANALYSIS",color=Color.White,fontSize=24.sp,fontWeight=FontWeight.Bold);Text(stageLabel(state.stage),color=Teal,fontSize=12.sp,letterSpacing=1.sp,modifier=Modifier.padding(top=4.dp))}
    state.bitmap?.let{b->item{Image(b.asImageBitmap(),"selected chart",Modifier.fillMaxWidth().height(170.dp),alignment=Alignment.Center)} }
    item{Button(onClick=pick,modifier=Modifier.fillMaxWidth().height(52.dp),shape=RoundedCornerShape(10.dp)){Text("SELECT SCREENSHOT")}}
    state.error?.let{item{InfoCard("ANALYSIS FAILED",it,"The file was rejected safely. Choose a PNG, JPEG, or WEBP chart screenshot and retry.")}}
    state.result?.let{r->item{ResultCard(r)}}
}}

private fun stageLabel(s:AnalysisStage)=when(s){AnalysisStage.Idle->"READY FOR INPUT";AnalysisStage.Validating->"VALIDATING SCREENSHOT";AnalysisStage.Reconstructing->"RECONSTRUCTING CHART";AnalysisStage.Extracting->"ANALYZING STRUCTURE + FEATURES";AnalysisStage.Ensemble->"RUNNING LOCAL MODEL ENSEMBLE";AnalysisStage.FinalDecision->"CALIBRATING + APPLYING ABSTENTION";AnalysisStage.Complete->"ANALYSIS COMPLETE"}

@Composable private fun ResultCard(r:ai.qxvision.signal.domain.AnalysisResult){val d=r.decision; val color=when(d.direction){ai.qxvision.signal.domain.SignalDirection.UP->Teal;ai.qxvision.signal.domain.SignalDirection.DOWN->Red;else->Color.White};Column(verticalArrangement=Arrangement.spacedBy(10.dp)){
    Surface(color=Panel,shape=RoundedCornerShape(16.dp),modifier=Modifier.fillMaxWidth()){Column(Modifier.padding(20.dp)){Text("NEXT 1M CANDLE",color=Muted,fontSize=11.sp,letterSpacing=2.sp);Text(d.direction.name.replace('_',' '),color=color,fontSize=38.sp,fontWeight=FontWeight.Bold,modifier=Modifier.padding(vertical=4.dp));Text(if(d.abstained)"NO TRADE IS A VALID DECISION" else "ACCEPTED BY POLICY",color=if(d.abstained) Muted else Teal,fontSize=12.sp,fontWeight=FontWeight.Bold);Spacer(Modifier.height(12.dp));StatLine("CALIBRATED PROBABILITY",d.calibratedProbability?.let{"%.1f%%".format(it*100)}?:"UNKNOWN");StatLine("CONFIDENCE",d.confidence?.let{"%.1f%%".format(it*100)}?:"UNKNOWN");StatLine("MODEL AGREEMENT",d.modelAgreement?.let{"%.1f%%".format(it*100)}?:"UNKNOWN");StatLine("UNCERTAINTY",d.uncertaintyScore?.let{"%.1f%%".format(it*100)}?:"UNKNOWN");StatLine("SCREENSHOT QUALITY","%.1f%%".format(d.screenshotQuality*100));StatLine("ANALYSIS TIME","${d.processingTimeMs} ms")}}
    EvidenceCard("EVIDENCE",d.bullishEvidence,d.bearishEvidence,d.neutralEvidence)
    InfoCard("REGIME",d.marketRegime.name.replace('_',' '),"Derived from normalized swing structure, trend efficiency, momentum, and volatility.")
    if(d.abstentionReasons.isNotEmpty())InfoCard("WHY NO TRADE",d.abstentionReasons.joinToString("  ·  "),"The policy never converts an unavailable or unreliable model into a prediction.")
    InfoCard("RUNNING CANDLE",if(r.sequence.runningCandleIndex!=null)"PARTIAL_OBSERVATION · CANDLE ${r.sequence.runningCandleIndex?.plus(1)}" else "UNKNOWN","The right-most visible candle is never assumed closed.")
}}

@Composable private fun EvidenceCard(title:String,bull:List<String>,bear:List<String>,neutral:List<String>){Surface(color=Panel,shape=RoundedCornerShape(12.dp),modifier=Modifier.fillMaxWidth()){Column(Modifier.padding(16.dp)){Text(title,color=Muted,fontSize=11.sp,letterSpacing=1.sp);if(bull.isNotEmpty())Text("BULLISH  ·  ${bull.joinToString("; ")}",color=Teal,fontSize=13.sp,modifier=Modifier.padding(top=8.dp));if(bear.isNotEmpty())Text("BEARISH  ·  ${bear.joinToString("; ")}",color=Red,fontSize=13.sp,modifier=Modifier.padding(top=6.dp));if(neutral.isNotEmpty())Text("NEUTRAL  ·  ${neutral.joinToString("; ")}",color=Muted,fontSize=13.sp,modifier=Modifier.padding(top=6.dp))}}}
@Composable private fun StatLine(a:String,b:String){Row(Modifier.fillMaxWidth().padding(vertical=4.dp),horizontalArrangement=Arrangement.SpaceBetween){Text(a,color=Muted,fontSize=12.sp);Text(b,color=Color.White,fontSize=12.sp,fontWeight=FontWeight.SemiBold)}}
@Composable private fun Metric(a:String,b:String,modifier:Modifier=Modifier){Surface(color=Panel,shape=RoundedCornerShape(10.dp),modifier=modifier){Column(Modifier.padding(12.dp)){Text(a,color=Muted,fontSize=9.sp);Text(b,color=Color.White,fontSize=12.sp,fontWeight=FontWeight.Bold)}}}
@Composable private fun InfoCard(title:String,value:String,detail:String){Surface(color=Panel,shape=RoundedCornerShape(12.dp),modifier=Modifier.fillMaxWidth()){Column(Modifier.padding(16.dp)){Text(title,color=Muted,fontSize=10.sp,letterSpacing=1.sp);Text(value,color=Color.White,fontSize=15.sp,fontWeight=FontWeight.Bold,modifier=Modifier.padding(top=6.dp));Text(detail,color=Muted,fontSize=12.sp,modifier=Modifier.padding(top=5.dp))}}}

@Composable private fun History(rows:List<PredictionEntity>,label:(String,String)->Unit){LazyColumn(verticalArrangement=Arrangement.spacedBy(10.dp),modifier=Modifier.fillMaxSize()){item{Text("HISTORY",color=Color.White,fontSize=24.sp,fontWeight=FontWeight.Bold);Text("Immutable prediction records · ${rows.size} samples",color=Muted,fontSize=13.sp,modifier=Modifier.padding(bottom=8.dp))};if(rows.isEmpty())item{InfoCard("NO RECORDS","UNKNOWN","Analyze a screenshot to create a local record. Incorrect predictions are never silently deleted.")}else items(rows){r->Surface(color=Panel,shape=RoundedCornerShape(12.dp),modifier=Modifier.fillMaxWidth()){Column(Modifier.padding(15.dp)){Row(Modifier.fillMaxWidth(),horizontalArrangement=Arrangement.SpaceBetween){Text(r.direction.replace('_',' '),color=if(r.direction=="UP")Teal else if(r.direction=="DOWN")Red else Color.White,fontWeight=FontWeight.Bold);Text(if(r.actualOutcome==null)"UNVERIFIED" else "ACTUAL ${r.actualOutcome}",color=Muted,fontSize=11.sp)}Text("${r.regime}  ·  ${r.modelVersion}",color=Muted,fontSize=11.sp,modifier=Modifier.padding(top=5.dp));Text("Quality ${"%.0f".format(r.quality*100)}%  ·  ${r.processingMs} ms",color=Muted,fontSize=11.sp);if(r.actualOutcome==null){Row(horizontalArrangement=Arrangement.spacedBy(6.dp),modifier=Modifier.horizontalScroll(rememberScrollState()).padding(top=8.dp)){TextButton(onClick={label(r.sampleId,"UP")}){Text("ACTUAL UP",color=Teal,fontSize=10.sp)};TextButton(onClick={label(r.sampleId,"DOWN")}){Text("ACTUAL DOWN",color=Red,fontSize=10.sp)};TextButton(onClick={label(r.sampleId,"DOJI")}){Text("DOJI",color=Muted,fontSize=10.sp)};TextButton(onClick={label(r.sampleId,"UNKNOWN")}){Text("UNKNOWN",color=Muted,fontSize=10.sp)}}}}}}}}

@Composable private fun Settings(current:String,set:(String)->Unit){LazyColumn(verticalArrangement=Arrangement.spacedBy(12.dp),modifier=Modifier.fillMaxSize()){item{Text("SETTINGS",color=Color.White,fontSize=24.sp,fontWeight=FontWeight.Bold);Text("Policy controls alter abstention thresholds only; they never guarantee a win rate.",color=Muted,fontSize=13.sp)};item{InfoCard("SIGNAL STRICTNESS",current,"Conservative settings require stronger quality, agreement, and probability. Custom remains bounded and still abstains when the artifact is unavailable.")};item{Row(horizontalArrangement=Arrangement.spacedBy(6.dp),modifier=Modifier.fillMaxWidth()){listOf("CONSERVATIVE","BALANCED","CUSTOM").forEach{mode->TextButton(onClick={set(mode)}){Text(mode,color=if(mode==current)Teal else Muted,fontSize=10.sp)}}}};item{InfoCard("MINIMUM MODEL INPUT","8–10 reliable candles","The running candle is represented separately and lowers reliability when unstable.")};item{InfoCard("PRIVACY","LOCAL ONLY","No screenshot upload, account credentials, advertising SDK, or automatic execution.")}}}

@Composable private fun Performance(rows:List<PredictionEntity>){val accepted=rows.count{!it.abstained};val abstained=rows.size-accepted;val verified=rows.filter{!it.abstained&&(it.actualOutcome=="UP"||it.actualOutcome=="DOWN")};val correct=verified.count{it.direction==it.actualOutcome};val accuracy=if(verified.isEmpty())null else correct.toFloat()/verified.size;LazyColumn(verticalArrangement=Arrangement.spacedBy(12.dp),modifier=Modifier.fillMaxSize()){item{Text("PERFORMANCE",color=Color.White,fontSize=24.sp,fontWeight=FontWeight.Bold);Text("Calculated only from local labelled records",color=Muted,fontSize=13.sp)};item{Row(horizontalArrangement=Arrangement.spacedBy(10.dp),modifier=Modifier.fillMaxWidth()){Metric("PREDICTIONS",rows.size.toString(),Modifier.weight(1f));Metric("ACCEPTED",accepted.toString(),Modifier.weight(1f));Metric("NO TRADE",abstained.toString(),Modifier.weight(1f))}};item{InfoCard("OBSERVED ACCURACY",accuracy?.let{"%.1f%%".format(it*100)}?:"UNKNOWN","${verified.size} verified directional outcomes. Unverified and NO TRADE rows are not silently treated as losses or wins.")};item{InfoCard("COVERAGE",if(rows.isEmpty())"UNKNOWN" else "%.1f%%".format(accepted.toFloat()/rows.size*100),"Accepted predictions divided by all screenshot analyses.")};item{InfoCard("CALIBRATION","UNKNOWN","Brier score, ECE, and confidence buckets become available after a validated model artifact and labelled history exist.")}}}

@Composable private fun ModelInfo(samples:Int){LazyColumn(verticalArrangement=Arrangement.spacedBy(12.dp),modifier=Modifier.fillMaxSize()){item{Text("MODEL",color=Color.White,fontSize=24.sp,fontWeight=FontWeight.Bold)};item{InfoCard("CURRENT MODEL","v1-untrained","No checked-in weights or fabricated validation metrics. A candidate artifact is created only by the chronological training gate.")};item{InfoCard("FEATURE SCHEMA","qxvision.features.v1","Normalized candle geometry, structure, trend, momentum, volatility, support/resistance, morphology, and explicitly tagged running-candle observation.")};item{InfoCard("ON-DEVICE RUNTIME","ONNX Runtime Mobile · READY FOR ARTIFACT","An ONNX session is reused and warmed when models/v1/ensemble/model.onnx is supplied. Missing, incompatible, or uncalibrated artifacts produce NO TRADE.")};item{InfoCard("DATASET","$samples local prediction records","Prediction history is not training data. Outcome labels remain separate and candidates require deduplication, leakage checks, walk-forward evaluation, calibration, and champion comparison.")};item{InfoCard("METRICS","UNKNOWN","Accuracy, Brier score, ECE, coverage, and latency are shown only when calculated from real records/artifacts.")}}}
