package pe.edu.ulima.paltascan

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import pe.edu.ulima.paltascan.ml.Analizador
import java.io.File

/**
 * Corre el Analizador sobre las fotos copiadas con adb a
 * <archivos externos de la app>/prueba/ y escribe resultados_app.json para
 * compararlo con scripts/referencia_app.py.
 */
@RunWith(AndroidJUnit4::class)
class ConcordanciaTest {

    @Test
    fun analizarFotosDePrueba() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "prueba")
        val fotos = dir.listFiles { f -> f.extension.lowercase() in listOf("jpg", "jpeg", "png") }?.sortedBy { it.name }.orEmpty()
        assertTrue("No hay fotos en $dir", fotos.isNotEmpty())

        val analizador = Analizador.cargar(ctx)
        val filas = fotos.map { f ->
            val r = analizador.analizar(BitmapFactory.decodeFile(f.absolutePath))
            """{"foto":"${f.name}","categoria":${r.categoria},"ratio":${"%.5f".format(java.util.Locale.US, r.ratio)},""" +
                """"madurez":${r.madurez},"prob_madurez":${r.probMadurez},"px_fruto":${r.fruto.count { it }},""" +
                """"px_defecto":${r.defecto.count { it }},"ms_pre":${r.tiempos.preprocesoMs},""" +
                """"ms_inf":${r.tiempos.inferenciaMs},"ms_post":${r.tiempos.postprocesoMs}}"""
        }
        analizador.cerrar()
        val json = "[\n" + filas.joinToString(",\n") + "\n]"
        File(ctx.getExternalFilesDir(null), "resultados_app.json").writeText(json)
        // Copia interna: algunas capas (MIUI) no dejan leer Android/data por adb;
        // esta se saca con `adb exec-out run-as pe.edu.ulima.paltascan cat files/resultados_app.json`
        File(ctx.filesDir, "resultados_app.json").writeText(json)
    }
}
