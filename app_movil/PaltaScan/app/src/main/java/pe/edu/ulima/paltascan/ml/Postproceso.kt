package pe.edu.ulima.paltascan.ml

import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Salidas del modelo ya separadas y en orden canal-mayor (elemento [c][i] en c * n + i).
 * Las cajas vienen en pixeles del lienzo (cx, cy, w, h).
 */
class SalidaModelo(
    val cajas: FloatArray,
    val puntajes: FloatArray,
    val coeficientes: FloatArray,
    val n: Int,
    val prototipos: FloatArray,
    val altoProto: Int,
    val anchoProto: Int,
    /** null en el YOLO sin cabezal de madurez (la madurez sale del clasificador). */
    val madurez: FloatArray?,
) {
    val canalesProto: Int get() = prototipos.size / (altoProto * anchoProto)
}

data class Instancia(
    val ancla: Int,
    val clase: Int,
    val confianza: Float,
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
)

/** Mascaras binarias en el espacio del contenido del letterbox (nh x nw). */
class Mascaras(val ancho: Int, val alto: Int, val palta: BooleanArray, val defecto: BooleanArray)

/** Port de inferencia_movil.postprocesar(). */
object Postproceso {

    fun seleccionar(
        s: SalidaModelo,
        confPalta: Float = Configuracion.CONF_PALTA,
        confDefecto: Float = Configuracion.CONF_DEFECTO,
        iou: Float = Configuracion.IOU_NMS,
        maxDet: Int = Configuracion.MAX_DETECCIONES,
    ): List<Instancia> {
        val n = s.n
        val candidatas = ArrayList<Instancia>()
        for (i in 0 until n) {
            val p0 = s.puntajes[i]
            val p1 = s.puntajes[n + i]
            val clase = if (p1 > p0) 1 else 0
            val conf = if (clase == 1) p1 else p0
            if (conf < (if (clase == 0) confPalta else confDefecto)) continue
            val cx = s.cajas[i]
            val cy = s.cajas[n + i]
            val w = s.cajas[2 * n + i]
            val h = s.cajas[3 * n + i]
            candidatas += Instancia(i, clase, conf, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        }
        return nmsPorClase(candidatas, iou).take(maxDet)
    }

    /** NMS por clase: igual que desplazar las cajas de cada clase en 4096 px. */
    fun nmsPorClase(instancias: List<Instancia>, iou: Float): List<Instancia> {
        val orden = instancias.sortedByDescending { it.confianza }
        val suprimida = BooleanArray(orden.size)
        val resultado = ArrayList<Instancia>()
        for (a in orden.indices) {
            if (suprimida[a]) continue
            val ia = orden[a]
            resultado += ia
            for (b in a + 1 until orden.size) {
                if (!suprimida[b] && orden[b].clase == ia.clase && iouCajas(ia, orden[b]) > iou) suprimida[b] = true
            }
        }
        return resultado
    }

    fun iouCajas(a: Instancia, b: Instancia): Float {
        val iw = max(0f, min(a.x2, b.x2) - max(a.x1, b.x1))
        val ih = max(0f, min(a.y2, b.y2) - max(a.y1, b.y1))
        val inter = iw * ih
        val union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
        return inter / (union + 1e-9f)
    }

    /**
     * Mascara por instancia = sigmoide(coeficientes . prototipos), reescalada
     * bilinealmente al lienzo (como cv2.resize INTER_LINEAR), recortada a su
     * caja y umbralizada en 0.5. Se devuelve solo la zona del contenido.
     */
    fun mascaras(s: SalidaModelo, instancias: List<Instancia>, lb: Letterbox): Mascaras {
        val palta = BooleanArray(lb.nh * lb.nw)
        val defecto = BooleanArray(lb.nh * lb.nw)
        val ph = s.altoProto
        val pw = s.anchoProto
        val c = s.canalesProto
        val escala = lb.lienzo.toFloat() / pw
        val prob = FloatArray(ph * pw)

        for (ins in instancias) {
            val bx1 = ins.x1.coerceIn(0f, lb.lienzo.toFloat()).toInt()
            val by1 = ins.y1.coerceIn(0f, lb.lienzo.toFloat()).toInt()
            val bx2 = ins.x2.coerceIn(0f, lb.lienzo.toFloat()).toInt()
            val by2 = ins.y2.coerceIn(0f, lb.lienzo.toFloat()).toInt()
            val x1 = max(bx1, lb.left)
            val y1 = max(by1, lb.top)
            val x2 = min(bx2, lb.left + lb.nw)
            val y2 = min(by2, lb.top + lb.nh)
            if (x1 >= x2 || y1 >= y2) continue

            val gx1 = max(0, floor((x1 + 0.5f) / escala - 0.5f).toInt())
            val gx2 = min(pw - 1, floor((x2 - 1 + 0.5f) / escala - 0.5f).toInt() + 1)
            val gy1 = max(0, floor((y1 + 0.5f) / escala - 0.5f).toInt())
            val gy2 = min(ph - 1, floor((y2 - 1 + 0.5f) / escala - 0.5f).toInt() + 1)
            for (gy in gy1..gy2) for (gx in gx1..gx2) {
                var z = 0f
                val base = gy * pw + gx
                for (k in 0 until c) z += s.coeficientes[k * s.n + ins.ancla] * s.prototipos[k * ph * pw + base]
                prob[base] = 1f / (1f + exp(-z))
            }

            val destino = if (ins.clase == 0) palta else defecto
            for (y in y1 until y2) {
                val (ya, yb, ty) = muestra(y, escala, ph)
                val fila = (y - lb.top) * lb.nw
                for (x in x1 until x2) {
                    val (xa, xb, tx) = muestra(x, escala, pw)
                    val arriba = prob[ya * pw + xa] * (1 - tx) + prob[ya * pw + xb] * tx
                    val abajo = prob[yb * pw + xa] * (1 - tx) + prob[yb * pw + xb] * tx
                    if (arriba * (1 - ty) + abajo * ty > 0.5f) destino[fila + (x - lb.left)] = true
                }
            }
        }
        return Mascaras(lb.nw, lb.nh, palta, defecto)
    }

    /**
     * U-Net (segmentacion semantica): la salida ya trae la probabilidad de palta
     * y de defecto por pixel del lienzo, [H, W, 2] (NHWC) o [2, H, W]. Se recorta
     * la zona del contenido y se umbraliza como eval_unet_multitarea.py
     * (palta > umbralPalta, defecto > umbralDefecto).
     */
    fun mascarasSemanticas(t: Tensor, lb: Letterbox, umbralPalta: Float, umbralDefecto: Float): Mascaras {
        val f = t.forma
        require(f.size == 3) { "Salida de mascaras con forma inesperada: ${f.joinToString()}" }
        val hwc = f[2] == 2
        val h = if (hwc) f[0] else f[1]
        val w = if (hwc) f[1] else f[2]
        require(h == lb.lienzo && w == lb.lienzo) { "Mascaras de ${h}x$w para un lienzo de ${lb.lienzo}" }
        val palta = BooleanArray(lb.nh * lb.nw)
        val defecto = BooleanArray(lb.nh * lb.nw)
        for (y in 0 until lb.nh) {
            val fila = (y + lb.top) * w
            for (x in 0 until lb.nw) {
                val p = fila + x + lb.left
                val pPalta = if (hwc) t.datos[2 * p] else t.datos[p]
                val pDefecto = if (hwc) t.datos[2 * p + 1] else t.datos[h * w + p]
                palta[y * lb.nw + x] = pPalta > umbralPalta
                defecto[y * lb.nw + x] = pDefecto > umbralDefecto
            }
        }
        return Mascaras(lb.nw, lb.nh, palta, defecto)
    }

    private data class Muestra(val a: Int, val b: Int, val t: Float)

    private fun muestra(d: Int, escala: Float, n: Int): Muestra {
        var f = (d + 0.5f) / escala - 0.5f
        if (f < 0f) f = 0f
        var a = floor(f).toInt()
        var t = f - a
        if (a >= n - 1) {
            a = n - 1
            t = 0f
        }
        return Muestra(a, min(a + 1, n - 1), t)
    }
}
