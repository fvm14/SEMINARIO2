package pe.edu.ulima.paltascan.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class DosCarasTest {

    private val p1 = floatArrayOf(0.7f, 0.2f, 0.1f, 0f, 0f)
    private val p2 = floatArrayOf(0.1f, 0.8f, 0.1f, 0f, 0f)

    @Test
    fun elDefectoSeSumaEnLasDosCaras() {
        // cada lado por separado seria Categoria I, pero el fruto completo no
        val c = DosCaras.combinar(0.06, p1, 0.05, p1)
        assertEquals(0, Ocde.clasificar(0.06))
        assertEquals(0, Ocde.clasificar(0.05))
        assertEquals(0.11, c.ratio, 1e-9)
        assertEquals(1, c.categoria)
        assertEquals(2, DosCaras.combinar(0.10, p1, 0.05, p1).categoria)
    }

    @Test
    fun limitesIgualQueDosCarasEvaluarPy() {
        assertEquals(0, DosCaras.combinar(0.045, p1, 0.045, p1).categoria) // 0.090
        assertEquals(1, DosCaras.combinar(0.07, p1, 0.07, p1).categoria) // 0.140
        assertEquals(2, DosCaras.combinar(0.0706, p1, 0.0706, p1).categoria) // 0.1412
    }

    @Test
    fun laMadurezEsElPromedioDeProbabilidades() {
        val c = DosCaras.combinar(0.0, p1, 0.0, p2)
        assertArrayEquals(floatArrayOf(0.4f, 0.5f, 0.1f, 0f, 0f), c.probabilidades, 1e-6f)
        assertEquals(2, c.madurez) // lado A decia 1 y lado B 2; el promedio decide
        assertEquals(0.5f, c.probMadurez, 1e-6f)
        assertEquals(1, DosCaras.combinar(0.0, p1, 0.0, p1).madurez)
    }
}
