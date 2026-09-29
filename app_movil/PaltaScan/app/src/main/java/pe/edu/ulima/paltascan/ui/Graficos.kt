package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.ml.Ocde

private fun Context.color(id: Int) = ContextCompat.getColor(this, id)
private fun View.dp(v: Float) = v * resources.displayMetrics.density

/**
 * Barra de "Area con defecto" (0 a 30 %) con las zonas OCDE y un marcador en
 * el valor medido; bajo la barra, las marcas 0 %, 9,4 %, 14,1 % y 30 %.
 */
class BarraDefectoView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    var ratio: Double = 0.0
        set(v) { field = v; invalidate() }

    private val maximo = 0.30
    private val pintura = Paint(Paint.ANTI_ALIAS_FLAG)
    private val texto = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = ctx.color(R.color.texto_suave)
        textSize = dp(11f)
    }

    override fun onMeasure(w: Int, h: Int) =
        setMeasuredDimension(MeasureSpec.getSize(w), dp(46f).toInt())

    override fun onDraw(c: Canvas) {
        val izq = dp(6f)
        val der = width - dp(6f)
        val ancho = der - izq
        val arriba = dp(8f)
        val abajo = dp(18f)
        fun x(r: Double) = izq + (ancho * (r / maximo)).toFloat().coerceIn(0f, ancho)
        val zonas = listOf(
            Triple(0.0, Ocde.LIMITE_CAT_I, R.color.cat_i),
            Triple(Ocde.LIMITE_CAT_I, Ocde.LIMITE_CAT_II, R.color.cat_ii),
            Triple(Ocde.LIMITE_CAT_II, maximo, R.color.rechazo),
        )
        for ((a, b, col) in zonas) {
            pintura.color = context.color(col)
            pintura.alpha = 70
            c.drawRect(x(a), arriba, x(b), abajo, pintura)
        }
        // marcador del valor
        val mx = x(ratio.coerceAtMost(maximo))
        pintura.alpha = 255
        pintura.color = context.color(R.color.texto)
        c.drawRoundRect(RectF(mx - dp(3f), arriba - dp(6f), mx + dp(3f), abajo + dp(6f)), dp(3f), dp(3f), pintura)
        // etiquetas
        val base = abajo + dp(20f)
        texto.textAlign = Paint.Align.LEFT
        c.drawText("0 %", izq, base, texto)
        texto.textAlign = Paint.Align.CENTER
        c.drawText("9,4 %", x(Ocde.LIMITE_CAT_I), base, texto)
        c.drawText("14,1 %", x(Ocde.LIMITE_CAT_II) + dp(8f), base, texto)
        texto.textAlign = Paint.Align.RIGHT
        c.drawText("30 %", der, base, texto)
    }
}

/** Escala de madurez 1-5: el nivel predicho resaltado, el resto tenue. */
class EscalaMadurezView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    var nivel: Int = 1
        set(v) { field = v; invalidate() }

    // De verde claro a casi negro, como la cascara de la palta Hass al madurar.
    private val tonos = intArrayOf(0xFF7CB342.toInt(), 0xFF558B2F.toInt(), 0xFF33691E.toInt(), 0xFF4E342E.toInt(), 0xFF2B1D17.toInt())
    private val pintura = Paint(Paint.ANTI_ALIAS_FLAG)
    private val texto = Paint(Paint.ANTI_ALIAS_FLAG).apply { textAlign = Paint.Align.CENTER; textSize = dp(13f); isFakeBoldText = true }

    override fun onMeasure(w: Int, h: Int) = setMeasuredDimension(MeasureSpec.getSize(w), dp(36f).toInt())

    override fun onDraw(c: Canvas) {
        val sep = dp(6f)
        val ancho = (width - 4 * sep) / 5
        for (i in 0 until 5) {
            val izq = i * (ancho + sep)
            val activo = i + 1 == nivel
            pintura.color = tonos[i]
            pintura.alpha = if (activo) 255 else 60
            c.drawRoundRect(RectF(izq, 0f, izq + ancho, height.toFloat()), dp(8f), dp(8f), pintura)
            texto.color = if (activo) 0xFFFFFFFF.toInt() else context.color(R.color.texto_suave)
            c.drawText("${i + 1}", izq + ancho / 2, height / 2 + dp(5f), texto)
        }
    }
}

/** Barra apilada con el % de Cat. I, Cat. II y Rechazado de un lote. */
class BarraCategoriasView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    private var conteos = IntArray(3)
    private val pintura = Paint(Paint.ANTI_ALIAS_FLAG)

    fun mostrar(porCategoria: IntArray) {
        conteos = porCategoria.copyOf()
        invalidate()
    }

    override fun onMeasure(w: Int, h: Int) = setMeasuredDimension(MeasureSpec.getSize(w), dp(10f).toInt())

    override fun onDraw(c: Canvas) {
        val total = conteos.sum()
        val r = RectF(0f, 0f, width.toFloat(), height.toFloat())
        pintura.color = context.color(R.color.borde)
        c.drawRoundRect(r, dp(5f), dp(5f), pintura)
        if (total == 0) return
        c.save()
        val clip = android.graphics.Path().apply { addRoundRect(r, dp(5f), dp(5f), android.graphics.Path.Direction.CW) }
        c.clipPath(clip)
        var x = 0f
        for (i in 0 until 3) {
            val w = width * conteos[i].toFloat() / total
            pintura.color = context.color(Textos.colorCategoria(i))
            c.drawRect(x, 0f, x + w, height.toFloat(), pintura)
            x += w
        }
        c.restore()
    }
}

/** Histograma de madurez 1-5 con el conteo encima de cada columna. */
class HistogramaMadurezView @JvmOverloads constructor(ctx: Context, attrs: AttributeSet? = null) : View(ctx, attrs) {
    private var conteos = IntArray(5)
    private val tonos = intArrayOf(0xFF7CB342.toInt(), 0xFF558B2F.toInt(), 0xFF33691E.toInt(), 0xFF4E342E.toInt(), 0xFF2B1D17.toInt())
    private val pintura = Paint(Paint.ANTI_ALIAS_FLAG)
    private val texto = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        textAlign = Paint.Align.CENTER
        textSize = dp(11f)
        color = ctx.color(R.color.texto_suave)
    }
    private val numero = Paint(texto).apply { isFakeBoldText = true; color = ctx.color(R.color.texto); textSize = dp(13f) }

    fun mostrar(porMadurez: IntArray) {
        conteos = porMadurez.copyOf()
        invalidate()
    }

    override fun onMeasure(w: Int, h: Int) = setMeasuredDimension(MeasureSpec.getSize(w), dp(150f).toInt())

    override fun onDraw(c: Canvas) {
        val sep = dp(10f)
        val ancho = (width - 4 * sep) / 5
        val pie = dp(44f)
        val techo = dp(20f)
        val alto = height - pie - techo
        val maximo = (conteos.maxOrNull() ?: 0).coerceAtLeast(1)
        for (i in 0 until 5) {
            val izq = i * (ancho + sep)
            val h = alto * conteos[i] / maximo
            pintura.color = tonos[i]
            c.drawRoundRect(RectF(izq, techo + alto - h, izq + ancho, techo + alto), dp(6f), dp(6f), pintura)
            c.drawText(conteos[i].toString(), izq + ancho / 2, techo + alto - h - dp(5f), numero)
            c.drawText("${i + 1}", izq + ancho / 2, height - pie + dp(16f), numero)
            val partes = Textos.NOMBRES_MADUREZ[i].split("-", " ")
            partes.take(2).forEachIndexed { k, p ->
                c.drawText(if (k == 0 && partes.size > 1 && Textos.NOMBRES_MADUREZ[i].contains("-")) "$p-" else p,
                    izq + ancho / 2, height - pie + dp(30f) + k * dp(12f), texto)
            }
        }
    }
}
