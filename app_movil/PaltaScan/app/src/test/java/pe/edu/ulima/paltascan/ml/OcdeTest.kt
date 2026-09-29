package pe.edu.ulima.paltascan.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class OcdeTest {

    @Test
    fun limitesDeCategoriaIgualQueOcdePy() {
        assertEquals(0, Ocde.clasificar(0.0))
        assertEquals(0, Ocde.clasificar(0.094))
        assertEquals(1, Ocde.clasificar(0.0941))
        assertEquals(1, Ocde.clasificar(0.141))
        assertEquals(2, Ocde.clasificar(0.1411))
    }

    @Test
    fun ratioSinFrutoEsCero() {
        assertEquals(0.0, Ocde.calcularRatio(BooleanArray(4), booleanArrayOf(true, false, false, false)), 0.0)
    }

    @Test
    fun frutoCompletoRellenaHuecosYDescartaManchasSueltas() {
        val w = 10
        val h = 10
        val palta = BooleanArray(w * h)
        for (y in 1..6) for (x in 1..6) palta[y * w + x] = true
        palta[3 * w + 3] = false
        palta[3 * w + 4] = false
        palta[9 * w + 9] = true
        val defecto = BooleanArray(w * h)
        defecto[3 * w + 3] = true

        val fruto = Ocde.frutoCompleto(palta, defecto, w, h)

        for (y in 1..6) for (x in 1..6) assertEquals("($x,$y)", true, fruto[y * w + x])
        assertEquals(false, fruto[9 * w + 9])
        assertEquals(36, fruto.count { it })
    }

    @Test
    fun dilatarIgualQueCv2ConAnclaCentral() {
        val w = 40
        val h = 30
        val m = BooleanArray(w * h)
        m[15 * w + 20] = true
        m[2 * w + 1] = true
        for (k in listOf(1, 2, 3, 20)) {
            assertArrayEquals("k=$k", dilatarFuerzaBruta(m, w, h, k), Ocde.dilatar(m, w, h, k))
        }
    }

    /** dst(x,y) = max src(x + i - a, y + j - a), i,j en [0,k), a = k/2 (cv2.dilate). */
    private fun dilatarFuerzaBruta(m: BooleanArray, w: Int, h: Int, k: Int): BooleanArray {
        val a = k / 2
        return BooleanArray(w * h) { p ->
            val x = p % w
            val y = p / w
            var hay = false
            for (j in 0 until k) for (i in 0 until k) {
                val sx = x + i - a
                val sy = y + j - a
                if (sx in 0 until w && sy in 0 until h && m[sy * w + sx]) hay = true
            }
            hay
        }
    }

    @Test
    fun roiEliminaDefectoLejosDelFruto() {
        val w = 50
        val h = 10
        val fruto = BooleanArray(w * h)
        for (x in 0..9) fruto[5 * w + x] = true
        val defecto = BooleanArray(w * h)
        defecto[5 * w + 15] = true
        defecto[5 * w + 45] = true
        val filtrado = Ocde.filtrarDefectoPorRoi(fruto, defecto, w, h, 20)
        assertEquals(true, filtrado[5 * w + 15])
        assertEquals(false, filtrado[5 * w + 45])
    }
}
