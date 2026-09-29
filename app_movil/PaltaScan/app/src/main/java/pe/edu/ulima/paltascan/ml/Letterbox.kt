package pe.edu.ulima.paltascan.ml

import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Geometria del letterbox de inferencia_movil.letterbox(): la foto se escala
 * manteniendo proporcion a nh x nw y se centra en un lienzo lienzo x lienzo.
 */
data class Letterbox(
    val lienzo: Int,
    val escala: Float,
    val top: Int,
    val left: Int,
    val nh: Int,
    val nw: Int,
) {
    companion object {
        fun calcular(alto: Int, ancho: Int, lienzo: Int): Letterbox {
            val r = min(lienzo.toDouble() / alto, lienzo.toDouble() / ancho)
            // kotlin.math.round redondea a par igual que round() de Python
            val nh = kotlin.math.round(alto * r).toInt()
            val nw = kotlin.math.round(ancho * r).toInt()
            return Letterbox(lienzo, r.toFloat(), (lienzo - nh) / 2, (lienzo - nw) / 2, nh, nw)
        }
    }

    fun kernelRoi(kernelOriginalPx: Int): Int = maxOf(1, (kernelOriginalPx * escala).roundToInt())
}
