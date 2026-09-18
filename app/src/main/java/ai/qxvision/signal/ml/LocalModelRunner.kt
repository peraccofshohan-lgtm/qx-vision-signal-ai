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
import java.security.MessageDigest
import org.json.JSONObject
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
    private var calibrationMethod = "platt"
    private var calibrationX = FloatArray(0)
    private var calibrationY = FloatArray(0)
    private var expectedFeatureCount = -1
    private var inputName = "features"
    private var outputName = "probability_up"

    init { load() }

    fun calibrated(raw: Float): Float {
        val p=raw.coerceIn(.000001f,.999999f)
        if (calibrationMethod == "isotonic" && calibrationX.isNotEmpty()) {
            if (p <= calibrationX.first()) return calibrationY.first().coerceIn(0f,1f)
            for (index in 1 until calibrationX.size) if (p <= calibrationX[index]) {
                val width=(calibrationX[index]-calibrationX[index-1]).coerceAtLeast(.000001f); val fraction=(p-calibrationX[index-1])/width
                return (calibrationY[index-1]+fraction*(calibrationY[index]-calibrationY[index-1])).coerceIn(0f,1f)
            }
            return calibrationY.last().coerceIn(0f,1f)
        }
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
            val metadata = context.assets.list("models/v1/ensemble")?.contains("metadata.json") == true
            if (!metadata) { failure = "MODEL_METADATA_UNAVAILABLE"; return }
            val metadataText = context.assets.open("models/v1/ensemble/metadata.json").bufferedReader().use { it.readText() }
            val expectedHash = Regex("\\\"onnx_sha256\\\"\\s*:\\s*\\\"([a-fA-F0-9]{64})\\\"").find(metadataText)?.groupValues?.get(1)
            val expectedSize = Regex("\\\"onnx_size_bytes\\\"\\s*:\\s*(\\d+)").find(metadataText)?.groupValues?.get(1)?.toLongOrNull()
            if (expectedHash == null || expectedSize == null || expectedSize != model.length() || !modelSha256(model).equals(expectedHash, ignoreCase=true)) { failure = "MODEL_CHECKSUM_MISMATCH"; return }
            session = environment.createSession(model.absolutePath, OrtSession.SessionOptions())
            if (!metadataText.contains("qxvision.features.v1")) { failure = "FEATURE_SCHEMA_MISMATCH"; session = null; return }
            expectedFeatureCount = Regex("\\\"feature_count\\\"\\s*:\\s*(\\d+)").find(metadataText)?.groupValues?.get(1)?.toIntOrNull() ?: -1
            inputName = Regex("\\\"input_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"").find(metadataText)?.groupValues?.get(1) ?: "features"
            outputName = Regex("\\\"output_name\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"").find(metadataText)?.groupValues?.get(1) ?: "probability_up"
            val schemaAsset = when {
                context.assets.list("models/v1/ensemble")?.contains("feature_schema.json") == true -> "models/v1/ensemble/feature_schema.json"
                context.assets.list("models/v1")?.contains("feature_schema.json") == true -> "models/v1/feature_schema.json"
                else -> null
            }
            if (schemaAsset == null) { failure = "FEATURE_SCHEMA_ARTIFACT_UNAVAILABLE"; session = null; return }
            val schemaText = context.assets.open(schemaAsset).bufferedReader().use { it.readText() }
            if (!schemaText.contains("qxvision.features.v1") || schemaText.count { it == '"' } < 10) { failure = "FEATURE_SCHEMA_INVALID"; session = null; return }
            if (!activeSessionHasExpectedIO(inputName, outputName)) { failure = "ONNX_IO_SCHEMA_MISMATCH"; session = null; return }
            modelVersion = Regex("\\\"model_version\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"").find(metadataText)?.groupValues?.get(1) ?: "UNKNOWN"
            val calibration = context.assets.list("models/v1/ensemble")?.contains("calibration.json") == true
            if (calibration) {
                val text = context.assets.open("models/v1/ensemble/calibration.json").bufferedReader().use { it.readText() }
                val calibrationJson=JSONObject(text); calibrationMethod=calibrationJson.optString("method","platt")
                if (calibrationMethod == "isotonic") {
                    val x=calibrationJson.optJSONArray("x"); val y=calibrationJson.optJSONArray("y")
                    if (x != null && y != null && x.length() == y.length()) { calibrationX=FloatArray(x.length()){x.optDouble(it).toFloat()}; calibrationY=FloatArray(y.length()){y.optDouble(it).toFloat()} }
                } else {
                    calibrationJson.optDouble("a",1.0).toFloat().let { calibrationA=it }; calibrationJson.optDouble("b",0.0).toFloat().let { calibrationB=it }
                }
                hasCalibration=calibrationJson.optBoolean("fitted",false) && (calibrationMethod != "isotonic" || calibrationX.isNotEmpty())
            }
        } catch (t: Throwable) {
            failure = "ONNX_LOAD_FAILED:${t.javaClass.simpleName}"
            session = null
        }
    }

    private fun modelSha256(file: File): String {
        val digest=MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input -> val buffer=ByteArray(8192); while(true){ val count=input.read(buffer); if(count<=0)break; digest.update(buffer,0,count) } }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun activeSessionHasExpectedIO(input: String, output: String): Boolean {
        val active = session ?: return false
        return expectedFeatureCount > 0 && active.inputNames.contains(input) && active.outputNames.contains(output)
    }

    fun warmup(featureCount: Int) { if (session != null) predict(FeatureVector(FloatArray(featureCount), List(featureCount) { "warmup_$it" }, FEATURE_SCHEMA_VERSION, BooleanArray(featureCount) { true }, emptyMap())) }

    fun predict(vector: FeatureVector): ModelPrediction {
        if (vector.schemaVersion != FEATURE_SCHEMA_VERSION) return ModelPrediction("onnx", .5f, false, "FEATURE_SCHEMA_MISMATCH")
        if (expectedFeatureCount > 0 && vector.values.size != expectedFeatureCount) return ModelPrediction("onnx", .5f, false, "ONNX_INPUT_SHAPE_MISMATCH")
        val active=session ?: return ModelPrediction("onnx", .5f, false, failure ?: "MODEL_ARTIFACT_UNAVAILABLE")
        return try {
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(vector.values), longArrayOf(1, vector.values.size.toLong())).use { tensor ->
                active.run(mapOf(inputName to tensor)).use { result ->
                    val raw=result[0].value
                    val value=when(raw) { is FloatArray -> raw.firstOrNull(); is Array<*> -> (raw.firstOrNull() as? FloatArray)?.firstOrNull(); is Float -> raw; is Double -> raw.toFloat(); else -> null } ?: throw IllegalStateException("unsupported ONNX output")
                    if (!value.isFinite() || value < 0f || value > 1f) throw IllegalStateException("invalid probability output")
                    val p=value
                    ModelPrediction("onnx",p,true)
                }
            }
        } catch (t: Throwable) { ModelPrediction("onnx", .5f, false, "ONNX_INFERENCE_FAILED:${t.javaClass.simpleName}") }
    }

    override fun close() { session?.close(); environment.close() }
}
