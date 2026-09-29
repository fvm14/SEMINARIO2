package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.datos.Inspeccion
import java.io.File

/** Acciones que se repiten en varias pantallas. */
object Acciones {

    /**
     * Lista de lotes para guardar o mover un analisis. La ultima opcion crea
     * un lote nuevo; `permitirQuitar` agrega "Quitar del lote".
     */
    fun elegirLote(
        context: Context, permitirQuitar: Boolean,
        alElegir: (loteId: Long?) -> Unit, alPedirNuevo: () -> Unit,
    ) {
        val lotes = context.app.baseDatos.lotes()
        val opciones = lotes.map { it.nombre }.toMutableList()
        if (permitirQuitar) opciones += context.getString(R.string.quitar_de_lote)
        opciones += "+ " + context.getString(R.string.nuevo_lote)
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.elegir_lote)
            .setItems(opciones.toTypedArray()) { _, i ->
                when {
                    i < lotes.size -> alElegir(lotes[i].id)
                    permitirQuitar && i == lotes.size -> alElegir(null)
                    else -> alPedirNuevo()
                }
            }
            .setNegativeButton(R.string.cancelar, null)
            .show()
    }

    fun confirmarEliminar(context: Context, i: Inspeccion, alEliminar: () -> Unit) {
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.eliminar_titulo)
            .setMessage(context.getString(R.string.eliminar_texto, Textos.fechaHora(i.fechaMs)))
            .setNegativeButton(R.string.cancelar, null)
            .setPositiveButton(R.string.eliminar) { _, _ ->
                eliminar(context, i)
                alEliminar()
            }
            .show()
    }

    /** Borra el registro y sus imagenes. */
    fun eliminar(context: Context, i: Inspeccion) {
        context.app.baseDatos.eliminar(i.id)
        listOf(i.archivoFoto, i.archivoOverlay).filter { it.isNotEmpty() }.forEach { File(it).delete() }
    }

    fun compartirArchivo(context: Context, archivo: File, mime: String, texto: String? = null) {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fotos", archivo)
        val envio = Intent(Intent.ACTION_SEND).apply {
            type = mime
            putExtra(Intent.EXTRA_STREAM, uri)
            texto?.let { putExtra(Intent.EXTRA_TEXT, it) }
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(envio, context.getString(R.string.compartir)))
    }

    /** Archivo temporal en cache/exportes para compartir un CSV o PDF. */
    fun archivoExporte(context: Context, nombre: String): File =
        File(File(context.cacheDir, "exportes").apply { mkdirs() }, nombre)
}
