package ai.qxvision.signal.vision

import android.graphics.Bitmap
import android.graphics.Color
import ai.qxvision.signal.domain.*
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/** Geometric, on-device screenshot parser. OCR is intentionally not a dependency. */
class ChartVisionEngine {
    private fun sat(c: Int): Float { val hi = max(Color.red(c), max(Color.green(c), Color.blue(c))); val lo = min(Color.red(c), min(Color.green(c), Color.blue(c))); return (hi - lo) / 255f }
    private fun lum(c: Int): Float = (0.2126f * Color.red(c) + 0.7152f * Color.green(c) + 0.0722f * Color.blue(c)) / 255f
    private fun dist(a: Int, b: Int): Float { val r=Color.red(a)-Color.red(b); val g=Color.green(a)-Color.green(b); val bl=Color.blue(a)-Color.blue(b); return sqrt((r*r+g*g+bl*bl).toFloat())/441.673f }
    private fun quant(c: Int): Int = Color.rgb(Color.red(c)/16*16+8, Color.green(c)/16*16+8, Color.blue(c)/16*16+8)
    private fun near(a: Int, b: Int?, tolerance: Float=.30f) = b != null && dist(a,b) <= tolerance

    fun screenshotHash(bitmap: Bitmap): String {
        val md=MessageDigest.getInstance("SHA-256"); md.update("${bitmap.width}x${bitmap.height}".toByteArray())
        val scaled=Bitmap.createScaledBitmap(bitmap, min(96,bitmap.width), min(96,bitmap.height), true)
        for (y in 0 until scaled.height) for (x in 0 until scaled.width) { val c=scaled.getPixel(x,y); md.update(byteArrayOf(Color.red(c).toByte(),Color.green(c).toByte(),Color.blue(c).toByte())) }
        return md.digest().joinToString("") { "%02x".format(it) }
    }

    private fun background(bitmap: Bitmap): Int {
        val counts=HashMap<Int,Int>(); val margin=max(1, min(bitmap.width,bitmap.height)/20)
        for (y in 0 until bitmap.height) for (x in 0 until bitmap.width) if (x<margin || y<margin || x>=bitmap.width-margin || y>=bitmap.height-margin) { val q=quant(bitmap.getPixel(x,y)); counts[q]=(counts[q]?:0)+1 }
        return counts.maxByOrNull { it.value }?.key ?: Color.BLACK
    }

    private fun candidates(bitmap: Bitmap, bounds: RectI, bg: Int): Pair<Int?,Int?> {
        val counts=HashMap<Int,Int>()
        for (y in bounds.top until bounds.bottom) for (x in bounds.left until bounds.right) { val c=bitmap.getPixel(x,y); if (sat(c)>.18f && dist(c,bg)>.07f) { val q=quant(c); counts[q]=(counts[q]?:0)+1 } }
        val ordered=counts.entries.sortedByDescending { it.value }.map { it.key }.distinct()
        val first=ordered.firstOrNull(); val second=ordered.drop(1).firstOrNull { dist(it,first?:it)>.23f }
        if (first==null) return null to null
        fun hue(c:Int):Float { val r=Color.red(c)/255f; val g=Color.green(c)/255f; val b=Color.blue(c)/255f; val hi=max(r,max(g,b)); val lo=min(r,min(g,b)); if(hi==lo)return 0f; val h=when(hi){r->(g-b)/(hi-lo);g->(b-r)/(hi-lo)+2;else->(r-g)/(hi-lo)+4}; return ((h/6f)%1f+1f)%1f }
        if(second==null) return first to null
        val h1=hue(first); val h2=hue(second)
        return if ((h1<.12f||h1>.9f) && h2 >= .20f && h2 <= .55f) second to first else if ((h2<.12f||h2>.9f) && h1 >= .20f && h1 <= .55f) first to second else first to second
    }

    fun inspect(bitmap: Bitmap, duplicate: Boolean=false): DetectedChart {
        val bg=background(bitmap); val bounds=RectI(0,0,bitmap.width,bitmap.height); val (bull,bear)=candidates(bitmap,bounds,bg)
        var saturated=0; var matched=0; var edge=0f; var count=0
        val stepX=max(1,bitmap.width/120); val stepY=max(1,bitmap.height/120)
        for(y in 0 until bitmap.height step stepY) for(x in 0 until bitmap.width step stepX){ val c=bitmap.getPixel(x,y); if(sat(c)>.18f && dist(c,bg)>.07f){saturated++; if(near(c,bull)||near(c,bear))matched++}; if(x+stepX<bitmap.width){edge+=abs(lum(bitmap.getPixel(x+stepX,y))-lum(c));count++} }
        val blur=(edge/max(1,count)*8f).coerceIn(0f,1f); val coverage=1f; val visible=(matched.toFloat()/max(1,saturated)).coerceIn(0f,1f); val resolution=(min(bitmap.width,bitmap.height)/240f).coerceIn(0f,1f)
        val marginX=max(2,bitmap.width/16); val marginY=max(2,bitmap.height/16); val rightLine=(0 until bitmap.width).takeLast(marginX).maxOfOrNull{x->(0 until bitmap.height).count{y->dist(bitmap.getPixel(x,y),bg)>.08f}.toFloat()/max(1,bitmap.height)}?:0f; val bottomLine=(0 until bitmap.height).takeLast(marginY).maxOfOrNull{y->(0 until bitmap.width).count{x->dist(bitmap.getPixel(x,y),bg)>.08f}.toFloat()/max(1,bitmap.width)}?:0f; val axis=max(rightLine,bottomLine).coerceIn(0f,1f); val indicator=((saturated-matched).toFloat()/max(1,count)*12f).coerceIn(0f,1f); val occlusion=(indicator*.65f+(1f-visible)*.35f).coerceIn(0f,1f)
        val reasons=mutableListOf<String>(); if(min(bitmap.width,bitmap.height)<160) reasons += "RESOLUTION_TOO_LOW"; if(bull==null||bear==null) reasons += "DIRECTION_COLORS_AMBIGUOUS"; if(visible<.25f) reasons += "CANDLE_PIXELS_NOT_RELIABLE"; if(blur<.035f) reasons += "LOW_EDGE_ENERGY_OR_BLUR"; if(duplicate) reasons += "DUPLICATE_SCREENSHOT"
        val score=(.30f*coverage+.25f*visible+.20f*resolution+.25f*blur).coerceIn(0f,1f)
        val supported=score>=.35f && !reasons.any { it=="RESOLUTION_TOO_LOW"||it=="DIRECTION_COLORS_AMBIGUOUS"||it=="CANDLE_PIXELS_NOT_RELIABLE" }
        return DetectedChart(bounds,bg,bull,bear,null,ScreenshotQuality(score,blur,coverage,visible,axis,indicator,occlusion,resolution,supported,reasons))
    }

    fun reconstruct(bitmap: Bitmap, chart: DetectedChart): CandleSequence {
        val b=chart.bounds; val active=BooleanArray(b.width)
        for(x in 0 until b.width){ var n=0; for(y in b.top until b.bottom){ val c=bitmap.getPixel(x,y); if(near(c,chart.bullishColor)||near(c,chart.bearishColor))n++ }; active[x]=n>=max(1,(b.height*.012f).toInt()) }
        val runs=mutableListOf<Pair<Int,Int>>(); var start=-1
        for(i in 0..active.size){ val on=i<active.size&&active[i]; if(on&&start<0)start=i; if(!on&&start>=0){if(i-start>=1)runs += (b.left+start) to (b.left+i); start=-1} }
        val candles=mutableListOf<ReconstructedCandle>()
        for((index,run) in runs.withIndex()){
            val colored=mutableListOf<Triple<Int,Int,Int>>(); for(x in run.first until run.second) for(y in b.top until b.bottom){val c=bitmap.getPixel(x,y);if(near(c,chart.bullishColor)||near(c,chart.bearishColor))colored += Triple(x,y,c)}
            if(colored.isEmpty())continue
            val ys=colored.map{it.second}; val top=ys.min(); val bottom=ys.max(); val row=colored.groupingBy{it.second}.eachCount(); val bodyRows=row.filter{it.value >= max(1,((run.second-run.first)*.45f).toInt())}.keys
            val bodyTop=bodyRows.minOrNull()?:top; val bodyBottom=bodyRows.maxOrNull()?:bottom; val bp=colored.count{near(it.third,chart.bullishColor)}; val rp=colored.size-bp; val dir=if(bp>rp)Direction.UP else if(rp>bp)Direction.DOWN else Direction.UNKNOWN
            val h=max(1,b.height).toFloat(); val high=1f-top/h; val low=1f-bottom/h; val bh=1f-bodyTop/h; val bl=1f-bodyBottom/h; val open=if(dir==Direction.UP)bl else if(dir==Direction.DOWN)bh else (bh+bl)/2f; val close=if(dir==Direction.UP)bh else if(dir==Direction.DOWN)bl else open; val total=max(1,bottom-top+1)/h; val body=max(0,bodyBottom-bodyTop+1)/h; val conf=(.35f+.45f*abs(bp-rp)/colored.size+.20f*((run.second-run.first)/5f).coerceAtMost(1f)).coerceAtMost(1f)
            candles += ReconstructedCandle(index,(run.first+run.second-1)/2f,bodyTop/h,bodyBottom/h,top/h,bottom/h,dir,body,(bodyTop-top)/h,(bottom-bodyBottom)/h,total,body/max(total,.0001f),open,close,high,low,true,conf,false)
        }
        val adjusted=candles.toMutableList(); if(adjusted.isNotEmpty()){val last=adjusted.last(); adjusted[adjusted.lastIndex]=last.copy(isComplete=false,partialObservation=true,visualConfidence=last.visualConfidence*.65f)}
        return CandleSequence(adjusted,chart,adjusted.lastIndex.takeIf{it>=0},adjusted.map{it.visualConfidence}.average().toFloat())
    }

    fun analyze(bitmap: Bitmap, duplicate: Boolean=false): CandleSequence { val c=inspect(bitmap,duplicate); return reconstruct(bitmap,c) }
}
