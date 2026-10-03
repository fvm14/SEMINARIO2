package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.view.View
import android.widget.EditText
import android.widget.LinearLayout
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.VistaResultadoBinding
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.datos.Lote
import pe.edu.ulima.paltascan.ml.Configuracion
import pe.edu.ulima.paltascan.ml.ValidacionCaptura.Problema
import java.io.File

/** Acciones y vistas que comparten la pantalla principal y el historial. */
object Acciones {

    /** Lista de lotes con "Sin lote" y "Nuevo lote". Devuelve el id elegido (null = sin lote). */
    fun elegirLote(context: Context, alElegir: (loteId: Long?) -> Unit) {
        val lotes = context.app.baseDatos.lotes()
        val opciones = listOf(context.getString(R.string.quitar_de_lote)) + lotes.map { it.nombre } +
            ("+ " + context.getString(R.string.nuevo_lote))
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.elegir_lote)
            .setItems(opciones.toTypedArray()) { _, i ->
                when (i) {
                    0 -> alElegir(null)
                    opciones.lastIndex -> crearLote(context, alElegir)
                    else -> alElegir(lotes[i - 1].id)
                }
            }
            .setNegativeButton(R.string.cancelar, null)
            .show()
    }

    private fun crearLote(context: Context, alCrear: (Long) -> Unit) {
        val dp = context.resources.displayMetrics.density
        val nombre = EditText(context).apply { hint = context.getString(R.string.nombre_lote) }
        val productor = EditText(context).apply { hint = context.getString(R.string.productor) }
        val campos = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding((20 * dp).toInt(), (8 * dp).toInt(), (20 * dp).toInt(), 0)
            addView(nombre)
            addView(productor)
        }
        MaterialAlertDialogBuilder(context)
            .setTitle(R.string.nuevo_lote)
            .setView(campos)
            .setNegativeButton(R.string.cancelar, null)
            .setPositiveButton(R.string.crear) { _, _ ->
                val n = nombre.text.toString().trim().ifEmpty { "Lote " + Textos.fechaHora(System.currentTimeMillis()) }
                val id = context.app.baseDatos.crearLote(
                    Lote(nombre = n, productor = productor.text.toString().trim(), notas = "", fechaMs = System.currentTimeMillis())
                )
                alCrear(id)
            }
            .show()
    }

    fun nombreLote(context: Context, loteId: Long?): String =
        loteId?.let { context.app.baseDatos.lote(it)?.nombre } ?: context.getString(R.string.sin_lote)

    /** Borra el registro y sus imagenes. */
    fun eliminar(context: Context, i: Inspeccion) {
        context.app.baseDatos.eliminar(i.id)
        listOf(i.archivoFoto, i.archivoOverlay).filter { it.isNotEmpty() }.forEach { File(it).delete() }
    }

    fun compartirArchivo(context: Context, archivo: File, mime: String) {
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.fotos", archivo)
        val envio = Intent(Intent.ACTION_SEND).apply {
            type = mime
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(envio, context.getString(R.string.compartir)))
    }

    fun archivoExporte(context: Context, nombre: String): File =
        File(File(context.cacheDir, "exportes").apply { mkdirs() }, nombre)
}

/** Llena la vista de resultado con un analisis guardado. */
fun VistaResultadoBinding.mostrar(i: Inspeccion) {
    val ctx = root.context
    val original = i.archivoFoto.takeIf { it.isNotEmpty() && File(it).exists() }
    fun pintar() {
        val ruta = if (interruptorMarcas.isChecked || original == null) i.archivoOverlay else original
        imagen.setImageBitmap(BitmapFactory.decodeFile(ruta))
    }
    interruptorMarcas.isEnabled = original != null
    interruptorMarcas.isChecked = true
    interruptorMarcas.setOnCheckedChangeListener { _, _ -> pintar() }
    pintar()

    textoCategoria.text = Textos.CATEGORIAS[i.categoria]
    textoCategoria.setTextColor(ContextCompat.getColor(ctx, Textos.colorCategoria(i.categoria)))
    textoRatio.text = Textos.porcentaje(i.ratio)
    textoMadurez.text = ctx.getString(
        R.string.madurez_valor, i.madurez, Textos.nombreMadurez(i.madurez), Math.round(i.probMadurez * 100)
    )
    textoTiempo.text = ctx.getString(R.string.tiempo_valor, i.msTotal / 1000)
    textoModelo.text = ctx.getString(R.string.modelo_valor, Configuracion.modelo(i.modelo).nombre)

    val avisos = i.avisos.split(",").filter { it.isNotBlank() }.mapNotNull { runCatching { Problema.valueOf(it) }.getOrNull() }
    textoAviso.visibility = if (avisos.isEmpty()) View.GONE else View.VISIBLE
    textoAviso.text = ctx.getString(R.string.aviso_poco_confiable) + " " + avisos.joinToString(", ") {
        ctx.getString(
            when (it) {
                Problema.OSCURA -> R.string.aviso_oscura
                Problema.BORROSA -> R.string.aviso_borrosa
                else -> R.string.aviso_confianza
            }
        )
    } + "."
}
