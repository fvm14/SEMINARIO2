package pe.edu.ulima.paltascan

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import pe.edu.ulima.paltascan.ml.Analizador
import pe.edu.ulima.paltascan.ml.Configuracion
import java.io.File

/**
 * Corre el Analizador sobre las fotos copiadas con adb a
 * <archivos externos de la app>/prueba/ con cada modelo instalado y escribe
 * resultados_app_<modelo>.json para compararlo con la referencia en Python
 * (referencia_app.py o eval_alternativos_movil.py).
 * Con -e modelo <id> se corre solo ese modelo.
 */
@RunWith(AndroidJUnit4::class)
class ConcordanciaTest {

    @Test
    fun analizarFotosDePrueba() {
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val dir = File(ctx.getExternalFilesDir(null), "prueba")
        val fotos = dir.listFiles { f -> f.extension.lowercase() in listOf("jpg", "jpeg", "png") }?.sortedBy { it.name }.orEmpty()
        assertTrue("No hay fotos en $dir", fotos.isNotEmpty())

        val soloModelo = InstrumentationRegistry.getArguments().getString("modelo")
        val modelos = Configuracion.MODELOS.filter { Analizador.existeModelo(ctx, it) && (soloModelo == null || it.id == soloModelo) }
        assertTrue("No hay modelos instalados", modelos.isNotEmpty())

        for (modelo in modelos) {
            val analizador = Analizador.cargar(ctx, modelo)
            val filas = fotos.map { f ->
                val r = analizador.analizar(BitmapFactory.decodeFile(f.absolutePath))
                """{"foto":"${f.name}","categoria":${r.categoria},"ratio":${"%.5f".format(java.util.Locale.US, r.ratio)},""" +
                    """"madurez":${r.madurez},"prob_madurez":${r.probMadurez},"px_fruto":${r.fruto.count { it }},""" +
                    """"px_defecto":${r.defecto.count { it }},"ms_pre":${r.tiempos.preprocesoMs},""" +
                    """"ms_inf":${r.tiempos.inferenciaMs},"ms_post":${r.tiempos.postprocesoMs}}"""
            }
            analizador.cerrar()
            val json = "[\n" + filas.joinToString(",\n") + "\n]"
            val nombre = if (modelo == Configuracion.MODELO_POR_DEFECTO) "resultados_app.json" else "resultados_app_${modelo.id}.json"
            File(ctx.getExternalFilesDir(null), nombre).writeText(json)
            // Copia interna: algunas capas (MIUI) no dejan leer Android/data por adb;
            // se saca con `adb exec-out run-as pe.edu.ulima.paltascan cat files/<nombre>`
            File(ctx.filesDir, nombre).writeText(json)
        }
    }
}
