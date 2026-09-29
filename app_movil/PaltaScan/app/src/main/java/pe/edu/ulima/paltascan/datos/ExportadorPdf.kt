package pe.edu.ulima.paltascan.datos

import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import pe.edu.ulima.paltascan.ml.Ocde
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Informe PDF de un lote (p. 29): resumen por categoria OCDE y madurez, y una
 * fila por analisis con su miniatura. Se genera en el celular con
 * android.graphics.pdf.PdfDocument (A4, 595 x 842 puntos).
 */
object ExportadorPdf {
    private const val ANCHO = 595
    private const val ALTO = 842
    private const val MARGEN = 40f
    private const val ALTO_FILA = 58f

    private val LOCALE = Locale.forLanguageTag("es-PE")
    private val NOMBRES_MADUREZ = listOf("Poco maduro", "Poco-medio", "Medio", "Medio-maduro", "Muy maduro")
    private val COLORES = intArrayOf(Color.rgb(46, 125, 50), Color.rgb(178, 106, 0), Color.rgb(198, 40, 40))
    private val FONDOS = intArrayOf(Color.rgb(227, 241, 225), Color.rgb(251, 239, 217), Color.rgb(251, 227, 225))

    private fun pintura(tam: Float, negrita: Boolean = false, color: Int = Color.rgb(28, 36, 25)) =
        Paint(Paint.ANTI_ALIAS_FLAG).apply {
            textSize = tam
            this.color = color
            typeface = Typeface.create(Typeface.DEFAULT, if (negrita) Typeface.BOLD else Typeface.NORMAL)
        }

    fun generar(lote: Lote, inspecciones: List<Inspeccion>, destino: File) {
        val doc = PdfDocument()
        val r = Resumen(inspecciones)
        val fecha = SimpleDateFormat("d MMM yyyy", LOCALE)
        val hora = SimpleDateFormat("d MMM · HH:mm", LOCALE)
        val suave = Color.rgb(94, 107, 88)
        var numPagina = 0
        lateinit var pagina: PdfDocument.Page
        lateinit var c: Canvas
        var y = 0f

        fun nuevaPagina() {
            if (numPagina > 0) doc.finishPage(pagina)
            numPagina++
            pagina = doc.startPage(PdfDocument.PageInfo.Builder(ANCHO, ALTO, numPagina).create())
            c = pagina.canvas
            y = MARGEN
            c.drawText("PaltaScan · informe de lote", MARGEN, ALTO - 20f, pintura(8f, color = suave))
            c.drawText("Página $numPagina", ANCHO - MARGEN - 40f, ALTO - 20f, pintura(8f, color = suave))
        }

        nuevaPagina()
        // Encabezado
        c.drawText(lote.nombre, MARGEN, y + 20f, pintura(20f, true)); y += 34f
        val origen = listOf(lote.productor, fecha.format(Date(lote.fechaMs))).filter { it.isNotBlank() }.joinToString(" · ")
        c.drawText(origen, MARGEN, y + 10f, pintura(11f, color = suave)); y += 22f
        c.drawText("${r.total} paltas analizadas", MARGEN, y + 12f, pintura(13f, true)); y += 26f
        if (lote.notas.isNotBlank()) {
            c.drawText("Notas: " + lote.notas.take(110), MARGEN, y + 10f, pintura(10f, color = suave)); y += 20f
        }

        // Tarjetas por categoria
        val anchoTarjeta = (ANCHO - 2 * MARGEN - 20f) / 3
        for (k in 0 until 3) {
            val x = MARGEN + k * (anchoTarjeta + 10f)
            val fondo = Paint().apply { color = FONDOS[k] }
            c.drawRoundRect(RectF(x, y, x + anchoTarjeta, y + 62f), 10f, 10f, fondo)
            c.drawText(listOf("Cat. I", "Cat. II", "Rechazado")[k], x + 12f, y + 18f, pintura(10f, color = COLORES[k]))
            c.drawText("${r.porcentaje(k)} %", x + 12f, y + 40f, pintura(18f, true, COLORES[k]))
            c.drawText("${r.porCategoria[k]} paltas", x + 12f, y + 54f, pintura(9f, color = COLORES[k]))
        }
        y += 80f

        // Madurez
        c.drawText("Madurez", MARGEN, y + 12f, pintura(12f, true)); y += 22f
        val anchoCol = (ANCHO - 2 * MARGEN) / 5
        val maximo = (r.porMadurez.maxOrNull() ?: 0).coerceAtLeast(1)
        for (k in 0 until 5) {
            val x = MARGEN + k * anchoCol
            val h = 50f * r.porMadurez[k] / maximo
            c.drawRect(x + 14f, y + 60f - h, x + anchoCol - 14f, y + 60f, Paint().apply { color = COLORES[0]; alpha = 90 + 30 * k })
            c.drawText(r.porMadurez[k].toString(), x + anchoCol / 2 - 6f, y + 56f - h, pintura(10f, true))
            c.drawText("${k + 1} · ${NOMBRES_MADUREZ[k]}", x + 4f, y + 74f, pintura(8f, color = suave))
        }
        y += 92f

        // Tabla de analisis
        fun cabeceraTabla() {
            val p = pintura(9f, true, suave)
            c.drawText("Foto", MARGEN, y + 10f, p)
            c.drawText("Fecha", MARGEN + 60f, y + 10f, p)
            c.drawText("Categoría OCDE", MARGEN + 160f, y + 10f, p)
            c.drawText("Madurez", MARGEN + 280f, y + 10f, p)
            c.drawText("Defecto", MARGEN + 400f, y + 10f, p)
            c.drawText("Tiempo", MARGEN + 460f, y + 10f, p)
            y += 16f
            c.drawLine(MARGEN, y, ANCHO - MARGEN, y, Paint().apply { color = Color.rgb(225, 228, 218) })
            y += 4f
        }
        c.drawText("Análisis del lote", MARGEN, y + 12f, pintura(12f, true)); y += 22f
        cabeceraTabla()
        for (i in inspecciones.sortedBy { it.fechaMs }) {
            if (y + ALTO_FILA > ALTO - 40f) {
                nuevaPagina()
                cabeceraTabla()
            }
            BitmapFactory.decodeFile(i.archivoOverlay, BitmapFactory.Options().apply { inSampleSize = 8 })?.let { bmp ->
                c.drawBitmap(bmp, null, RectF(MARGEN, y + 3f, MARGEN + 50f, y + 53f), null)
                bmp.recycle()
            }
            val base = y + 30f
            c.drawText(hora.format(Date(i.fechaMs)), MARGEN + 60f, base, pintura(10f))
            c.drawText(Ocde.CATEGORIAS[i.categoria], MARGEN + 160f, base, pintura(10f, true, COLORES[i.categoria]))
            c.drawText("${i.madurez} · ${NOMBRES_MADUREZ[i.madurez - 1]}", MARGEN + 280f, base, pintura(10f))
            c.drawText("%.1f %%".format(LOCALE, i.ratio * 100), MARGEN + 400f, base, pintura(10f))
            c.drawText("%.1f s".format(LOCALE, i.msTotal / 1000), MARGEN + 460f, base, pintura(10f, color = suave))
            y += ALTO_FILA
        }
        doc.finishPage(pagina)
        FileOutputStream(destino).use { doc.writeTo(it) }
        doc.close()
    }
}
