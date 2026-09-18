package ai.qxvision.signal.data

import ai.qxvision.signal.domain.AnalysisResult
import kotlinx.coroutines.flow.Flow
import java.util.UUID

class PredictionRepository(private val dao:PredictionDao){
    val history:Flow<List<PredictionEntity>> = dao.observe()
    suspend fun save(result:AnalysisResult){ val d=result.decision; dao.insert(PredictionEntity(UUID.randomUUID().toString(),System.currentTimeMillis(),result.screenshotHash,d.direction.name,d.calibratedProbability,d.confidence,d.abstained,d.abstentionReasons.joinToString("|"),d.marketRegime.name,d.screenshotQuality,d.processingTimeMs,d.modelVersion)) }
    suspend fun label(id:String,outcome:String){require(outcome in setOf("UP","DOWN","INVALID_UNKNOWN"));dao.label(id,outcome)}
}
