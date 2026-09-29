package pe.edu.ulima.paltascan.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import pe.edu.ulima.paltascan.ml.ValidacionCaptura.Problema

class ValidacionCapturaTest {

    private fun disco(ancho: Int, alto: Int, cx: Int, cy: Int, r: Int, m: BooleanArray = BooleanArray(ancho * alto)): BooleanArray {
        for (y in 0 until alto) for (x in 0 until ancho) {
            if ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r) m[y * ancho + x] = true
        }
        return m
    }

    @Test
    fun frutoCentradoYFotoBuenaNoTieneProblemas() {
        val m = disco(100, 100, 50, 50, 30)
        val p = ValidacionCaptura.revisar(m, 100, 100, ValidacionCaptura.contarFrutos(m, 100, 100), 200.0, 200.0, 0.9f)
        assertTrue(p.isEmpty())
    }

    @Test
    fun sinFrutoSoloReportaSinPalta() {
        val p = ValidacionCaptura.revisar(BooleanArray(100), 10, 10, 0, 10.0, 0.0, 0.1f)
        assertEquals(listOf(Problema.SIN_PALTA), p)
    }

    @Test
    fun cuentaDosFrutosSeparados() {
        val m = disco(200, 100, 50, 50, 30)
        disco(200, 100, 150, 50, 28, m)
        assertEquals(2, ValidacionCaptura.contarFrutos(m, 200, 100))
    }

    @Test
    fun ignoraManchasPequenasAlContarFrutos() {
        val m = disco(200, 100, 50, 50, 30)
        disco(200, 100, 170, 20, 3, m)
        assertEquals(1, ValidacionCaptura.contarFrutos(m, 200, 100))
    }

    @Test
    fun detectaFrutoCortadoEnElBorde() {
        val m = disco(100, 100, 5, 50, 30)
        assertTrue(ValidacionCaptura.tocaBorde(m, 100, 100))
        assertFalse(ValidacionCaptura.tocaBorde(disco(100, 100, 50, 50, 30), 100, 100))
    }

    @Test
    fun avisosNoBloqueanElAnalisis() {
        val m = disco(100, 100, 50, 50, 30)
        val p = ValidacionCaptura.revisar(m, 100, 100, 1, 30.0, 5.0, 0.4f)
        assertEquals(listOf(Problema.OSCURA, Problema.BORROSA, Problema.BAJA_CONFIANZA), p)
        assertTrue(p.none { it.bloqueante })
    }

    @Test
    fun nitidezDeImagenPlanaEsCeroYDeTableroEsAlta() {
        val plana = IntArray(64) { 128 }
        assertEquals(0.0, ValidacionCaptura.nitidez(plana, 8, 8), 1e-9)
        val tablero = IntArray(64) { i -> if ((i % 8 + i / 8) % 2 == 0) 0 else 255 }
        assertTrue(ValidacionCaptura.nitidez(tablero, 8, 8) > 1000)
    }
}
