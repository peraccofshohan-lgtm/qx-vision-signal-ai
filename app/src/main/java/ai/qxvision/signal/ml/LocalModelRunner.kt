package ai.qxvision.signal.ml

import android.content.Context
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.qxvision.signal.domain.FeatureVector
import ai.qxvision.signal.domain.ModelPrediction
import ai.qxvision.signal.features.FEATURE_SCHEMA_VERSION
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.exp

/** Loads only a repository-provided, schema-matched ONNX artifact. */
class LocalModelRunner(private val context: Context) : AutoCloseable {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    var modelVersion: String = "v1-untrained"
        private set
    var failure: String? = null
        private set
    var hasCalibration: Boolean = false
        private set
    private var calibrationA = 1f
    private var calibrationB = 0f

    init { load() }

    fun calibrated(raw: Float): Float {
        val p=raw.coerceIn(.000001f,.999999f)
        val logit=kotlin.math.ln(p/(1f-p))
        return (1f/(1f+exp(-(calibrationA*logit+calibrationB)))).coerceIn(0f,1f)
    }

    private fun load() {
        try {
            val assetPath = "models/v1/ensemble/model.onnx"
            val info = context.assets.list("models/v1/ensemble")?.toSet().orEmpty()
            if ("model.onnx" !in info) { failure = "MODEL_ARTIFACT_UNAVAILABLE"; return }
            val model = File(context.codeCacheDir, "model.onnx")
            context.assets.open(assetPath).use { input -> model.outputStream().use { input.copyTo(it) } }
            session = environment.createSession(model.absolutePath, OrtSession.SessionOptions())
            val metadata = context.assets.list("models/v1/ensemble")?.contains("metadata.json") == true
            if (!metadata) { failure = "MODEL_METADATA_UNAVAILABLE"; session = null; return }
            val metadataText = context.assets.open("models/v1/ensemble/metadata.json").bufferedReader().use { it.readText() }
            if (!metadataText.contains("qxvision.features.v1")) { failure = "FEATURE_SCHEMA_MISMATCH"; session = null; return }
            modelVersion = Regex("\\\"model_version\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"").find(metadataText)?.groupValues?.get(1) ?: "UNKNOWN"
            val calibration = context.assets.list("models/v1/ensemble")?.contains("calibration.json") == true
            if (calibration) {
                val text = context.assets.open("models/v1/ensemble/calibration.json").bufferedReader().use { it.readText() }
                Regex("\\\"a\\\"\\s*:\\s*([-+0-9.eE]+)").find(text)?.groupValues?.get(1)?.toFloatOrNull()?.let { calibrationA = it }
                Regex("\\\"b\\\"\\s*:\\s*([-+0-9.eE]+)").find(text)?.groupValues?.get(1)?.toFloatOrNull()?.let { calibrationB = it }
                hasCalibration = Regex("\\\"fitted\\\"\\s*:\\s*true").containsMatchIn(text)
            }
        } catch (t: Throwable) {
            failure = "ONNX_LOAD_FAILED:${t.javaClass.simpleName}"
            session = null
        }
    }

    fun warmup(featureCount: Int) { if (session != null) predict(FeatureVector(FloatArray(featureCount), List(featureCount) { "warmup_$it" }, FEATURE_SCHEMA_VERSION, BooleanArray(featureCount) { true }, emptyMap())) }

    fun predict(vector: FeatureVector): ModelPrediction {
        if (vector.schemaVersion != FEATURE_SCHEMA_VERSION) return ModelPrediction("onnx", .5f, false, "FEATURE_SCHEMA_MISMATCH")
        val active=session ?: return ModelPrediction("onnx", .5f, false, failure ?: "MODEL_ARTIFACT_UNAVAILABLE")
        return try {
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(vector.values), longArrayOf(1, vector.values.size.toLong())).use { tensor ->
                active.run(mapOf(active.inputNames.first() to tensor)).use { result ->
                    val raw=result[0].value
                    val value=when(raw) { is FloatArray -> raw.firstOrNull(); is Array<*> -> (raw.firstOrNull() as? FloatArray)?.firstOrNull(); is Float -> raw; is Double -> raw.toFloat(); else -> null } ?: throw IllegalStateException("unsupported ONNX output")
                    val p=value.coerceIn(0f,1f)
                    ModelPrediction("onnx",p,true)
                }
            }
        } catch (t: Throwable) { ModelPrediction("onnx", .5f, false, "ONNX_INFERENCE_FAILED:${t.javaClass.simpleName}") }
    }

    override fun close() { session?.close(); environment.close() }
}
