package ai.qxvision.signal.data

import android.content.Context
import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Entity(tableName="predictions")
data class PredictionEntity(
    @PrimaryKey val sampleId:String,
    val createdAtUtc:Long,
    val screenshotHash:String,
    val direction:String,
    val probability:Float?,
    val confidence:Float?,
    val abstained:Boolean,
    val reasons:String,
    val regime:String,
    val quality:Float,
    val processingMs:Long,
    val modelVersion:String,
    val actualOutcome:String?=null,
    val notes:String?=null
)

@Dao
interface PredictionDao {
    @Insert(onConflict=OnConflictStrategy.ABORT) suspend fun insert(row:PredictionEntity)
    @Query("SELECT * FROM predictions ORDER BY createdAtUtc DESC") fun observe():Flow<List<PredictionEntity>>
    @Query("UPDATE predictions SET actualOutcome=:outcome WHERE sampleId=:id AND actualOutcome IS NULL") suspend fun label(id:String,outcome:String)
    @Query("SELECT COUNT(*) FROM predictions") suspend fun count():Int
}

@Database(entities=[PredictionEntity::class],version=1,exportSchema=false)
abstract class SignalDatabase:RoomDatabase(){ abstract fun predictions():PredictionDao
    companion object { @Volatile private var instance:SignalDatabase?=null; fun get(context:Context)=instance?: synchronized(this){instance?:Room.databaseBuilder(context.applicationContext,SignalDatabase::class.java,"qx_signal.db").build().also{instance=it}} }
}
