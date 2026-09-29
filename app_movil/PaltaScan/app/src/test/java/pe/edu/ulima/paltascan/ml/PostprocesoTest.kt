package pe.edu.ulima.paltascan.ml

import org.junit.Assert.assertEquals
import org.junit.Test

class PostprocesoTest {

    @Test
    fun letterboxIgualQuePython() {
        val cuadrada = Letterbox.calcular(800, 800, 800)
        assertEquals(Letterbox(800, 1f, 0, 0, 800, 800), cuadrada)
        val horizontal = Letterbox.calcular(1200, 1600, 800)
        assertEquals(600, horizontal.nh)
        assertEquals(800, horizontal.nw)
        assertEquals(100, horizontal.top)
        assertEquals(0, horizontal.left)
        assertEquals(10, horizontal.kernelRoi(20))
    }

    @Test
    fun nmsSuprimeSoloDentroDeLaMismaClase() {
        val a = Instancia(0, 0, 0.9f, 0f, 0f, 100f, 100f)
        val b = Instancia(1, 0, 0.8f, 5f, 5f, 100f, 100f)
        val c = Instancia(2, 1, 0.7f, 5f, 5f, 100f, 100f)
        val d = Instancia(3, 0, 0.6f, 200f, 200f, 300f, 300f)
        val r = Postproceso.nmsPorClase(listOf(d, c, b, a), 0.7f)
        assertEquals(listOf(0, 2, 3), r.map { it.ancla })
    }

    /** 1 ancla de palta con caja [8,16) x [4,12) en un lienzo de 32 px y prototipos de 8x8. */
    private fun salidaSintetica(confPalta: Float, signoProto: Float): SalidaModelo {
        val n = 1
        val c = 32
        val ph = 8
        val pw = 8
        val coef = FloatArray(c * n).also { it[0] = 1f }
        val proto = FloatArray(c * ph * pw).also { p -> for (i in 0 until ph * pw) p[i] = 10f * signoProto }
        return SalidaModelo(
            cajas = floatArrayOf(12f, 8f, 8f, 8f),
            puntajes = floatArrayOf(confPalta, 0.01f),
            coeficientes = coef, n = n,
            prototipos = proto, altoProto = ph, anchoProto = pw,
            madurez = floatArrayOf(0.1f, 0.1f, 0.6f, 0.1f, 0.1f),
        )
    }

    @Test
    fun mascaraQuedaRecortadaASuCaja() {
        val s = salidaSintetica(0.9f, +1f)
        val lb = Letterbox.calcular(32, 32, 32)
        val ins = Postproceso.seleccionar(s, 0.35f, 0.05f)
        assertEquals(1, ins.size)
        val m = Postproceso.mascaras(s, ins, lb)
        for (y in 0 until 32) for (x in 0 until 32) {
            val dentro = x in 8 until 16 && y in 4 until 12
            assertEquals("($x,$y)", dentro, m.palta[y * 32 + x])
        }
        assertEquals(0, m.defecto.count { it })
    }

    @Test
    fun probabilidadBajaNoGeneraMascara() {
        val s = salidaSintetica(0.9f, -1f)
        val m = Postproceso.mascaras(s, Postproceso.seleccionar(s), Letterbox.calcular(32, 32, 32))
        assertEquals(0, m.palta.count { it })
    }

    @Test
    fun confianzaBajoUmbralSeDescarta() {
        assertEquals(0, Postproceso.seleccionar(salidaSintetica(0.2f, 1f), 0.35f, 0.05f).size)
    }

    @Test
    fun mascaraSoloEnZonaDeContenidoDelLetterbox() {
        // Foto 16x32 (alto x ancho) en lienzo 32: contenido en filas 8..23
        val s = salidaSintetica(0.9f, +1f)
        val lb = Letterbox.calcular(16, 32, 32)
        assertEquals(8, lb.top)
        val m = Postproceso.mascaras(s, Postproceso.seleccionar(s), lb)
        assertEquals(32, m.ancho)
        assertEquals(16, m.alto)
        // caja y en [4,12) -> en contenido solo filas 8..11 -> y' 0..3
        for (y in 0 until 16) for (x in 0 until 32) {
            assertEquals("($x,$y)", x in 8 until 16 && y in 0 until 4, m.palta[y * 32 + x])
        }
    }
}
