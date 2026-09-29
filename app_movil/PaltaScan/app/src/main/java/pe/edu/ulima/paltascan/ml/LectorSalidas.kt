package pe.edu.ulima.paltascan.ml

/** Un tensor de salida ya descuantizado, sin la dimension de lote. */
class Tensor(val forma: IntArray, val datos: FloatArray)

/**
 * Identifica las salidas por su forma, como BackendTFLite.__call__ en
 * inferencia_movil.py (el orden de salida de onnx2tf no es fijo):
 *   5 valores           -> madurez
 *   3D                  -> prototipos ([H, W, 32] o [32, H, W])
 *   2D [C, N] o [N, C]  -> cajas (C=4), puntajes (C=2), coeficientes (C=32)
 *                          o el formato antiguo con todo junto (C=38)
 */
object LectorSalidas {

    fun leer(tensores: List<Tensor>, lienzo: Int): SalidaModelo {
        var madurez: FloatArray? = null
        var proto: FloatArray? = null
        var ph = 0
        var pw = 0
        val partes = HashMap<Int, FloatArray>()
        var n = 0

        for (t in tensores) {
            val f = t.forma
            when {
                t.datos.size == 5 -> madurez = t.datos
                f.size == 3 -> {
                    if (f[2] == 32) {
                        ph = f[0]; pw = f[1]
                        proto = transponerHwcAChw(t.datos, ph, pw, 32)
                    } else {
                        ph = f[1]; pw = f[2]
                        proto = t.datos
                    }
                }
                f.size == 2 -> {
                    val (c, datos) = if (f[0] < f[1]) f[0] to t.datos else f[1] to transponer(t.datos, f[0], f[1])
                    n = maxOf(f[0], f[1])
                    partes[c] = datos
                }
                else -> error("Salida con forma inesperada: ${f.joinToString()}")
            }
        }

        val cajas: FloatArray
        val puntajes: FloatArray
        val coefs: FloatArray
        val junto = partes[38]
        if (junto != null) {
            cajas = junto.copyOfRange(0, 4 * n)
            puntajes = junto.copyOfRange(4 * n, 6 * n)
            coefs = junto.copyOfRange(6 * n, 38 * n)
        } else {
            cajas = requireNotNull(partes[4]) { "Falta la salida de cajas" }.let { a -> FloatArray(a.size) { a[it] * lienzo } }
            puntajes = requireNotNull(partes[2]) { "Falta la salida de puntajes" }
            coefs = requireNotNull(partes[32]) { "Falta la salida de coeficientes" }
        }
        return SalidaModelo(
            cajas, puntajes, coefs, n,
            requireNotNull(proto) { "Falta la salida de prototipos" }, ph, pw,
            requireNotNull(madurez) { "Falta la salida de madurez" },
        )
    }

    /** [filas, columnas] -> [columnas, filas]. */
    fun transponer(a: FloatArray, filas: Int, columnas: Int): FloatArray {
        val out = FloatArray(a.size)
        for (i in 0 until filas) for (j in 0 until columnas) out[j * filas + i] = a[i * columnas + j]
        return out
    }

    fun transponerHwcAChw(a: FloatArray, h: Int, w: Int, c: Int): FloatArray {
        val out = FloatArray(a.size)
        for (p in 0 until h * w) for (k in 0 until c) out[k * h * w + p] = a[p * c + k]
        return out
    }
}
