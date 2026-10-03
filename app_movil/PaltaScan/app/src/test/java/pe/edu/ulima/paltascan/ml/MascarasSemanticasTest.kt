package pe.edu.ulima.paltascan.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

/** Mascaras de la U-Net: recorte del contenido del letterbox y umbrales de eval_unet_multitarea.py. */
class MascarasSemanticasTest {

    // Lienzo de 4 x 4 con una foto de 4 x 2 (alto x ancho): columnas 1 y 2 son contenido
    private val lb = Letterbox.calcular(alto = 4, ancho = 2, lienzo = 4)
    private val palta = FloatArray(16) { if (it % 4 in 1..2) 0.9f else 0.2f }
    private val defecto = FloatArray(16) { i -> if (i == 5) 0.3f else if (i == 6) 0.05f else 0f }

    @Test
    fun recortaElContenidoYUmbraliza() {
        assertEquals(1, lb.left)
        assertEquals(2, lb.nw)
        val hwc = FloatArray(32) { i -> if (i % 2 == 0) palta[i / 2] else defecto[i / 2] }
        val m = Postproceso.mascarasSemanticas(Tensor(intArrayOf(4, 4, 2), hwc), lb, 0.5f, 0.10f)
        assertEquals(2, m.ancho)
        assertEquals(4, m.alto)
        assertArrayEquals(BooleanArray(8) { true }, m.palta)
        // pixel 5 del lienzo = fila 1, columna 1 -> contenido (1, 0) = indice 2; 0.05 no supera 0.10
        assertArrayEquals(BooleanArray(8) { it == 2 }, m.defecto)
    }

    @Test
    fun aceptaCanalesPrimero() {
        val chw = palta + defecto
        val hwc = FloatArray(32) { i -> if (i % 2 == 0) palta[i / 2] else defecto[i / 2] }
        val a = Postproceso.mascarasSemanticas(Tensor(intArrayOf(2, 4, 4), chw), lb, 0.5f, 0.10f)
        val b = Postproceso.mascarasSemanticas(Tensor(intArrayOf(4, 4, 2), hwc), lb, 0.5f, 0.10f)
        assertArrayEquals(b.palta, a.palta)
        assertArrayEquals(b.defecto, a.defecto)
    }
}
