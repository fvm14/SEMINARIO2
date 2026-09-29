package pe.edu.ulima.paltascan.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class LectorSalidasTest {

    private val n = 40
    private val lienzo = 16

    private fun serie(tamano: Int, base: Float) = FloatArray(tamano) { base + it }

    @Test
    fun reconoceSalidasPorFormaEnCualquierOrden() {
        val cajas = serie(4 * n, 0f).map { it / 1000f }.toFloatArray()
        val punt = serie(2 * n, 0f)
        val coef = serie(32 * n, 0f)
        val protoChw = serie(32 * 4 * 4, 0f)
        val protoHwc = LectorSalidas.transponer(protoChw, 32, 16)
        val mad = floatArrayOf(0.1f, 0.2f, 0.4f, 0.2f, 0.1f)

        val s = LectorSalidas.leer(
            listOf(
                Tensor(intArrayOf(5), mad),
                Tensor(intArrayOf(n, 32), LectorSalidas.transponer(coef, 32, n)),
                Tensor(intArrayOf(4, 4, 32), protoHwc),
                Tensor(intArrayOf(2, n), punt),
                Tensor(intArrayOf(n, 4), LectorSalidas.transponer(cajas, 4, n)),
            ),
            lienzo,
        )

        assertEquals(n, s.n)
        assertArrayEquals(cajas.map { it * lienzo }.toFloatArray(), s.cajas, 1e-5f)
        assertArrayEquals(punt, s.puntajes, 0f)
        assertArrayEquals(coef, s.coeficientes, 0f)
        assertArrayEquals(protoChw, s.prototipos, 0f)
        assertEquals(4, s.altoProto)
        assertEquals(4, s.anchoProto)
        assertArrayEquals(mad, s.madurez, 0f)
    }

    @Test
    fun formatoAntiguoConDeteccionesJuntas() {
        val det = serie(38 * n, 0f)
        val s = LectorSalidas.leer(
            listOf(
                Tensor(intArrayOf(38, n), det),
                Tensor(intArrayOf(32, 4, 4), serie(512, 0f)),
                Tensor(intArrayOf(5), FloatArray(5)),
            ),
            lienzo,
        )
        assertArrayEquals(det.copyOfRange(0, 4 * n), s.cajas, 0f)
        assertArrayEquals(det.copyOfRange(6 * n, 38 * n), s.coeficientes, 0f)
    }
}
