package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.provider.OpenableColumns
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.databinding.ActivityVistaPreviaBinding
import pe.edu.ulima.paltascan.datos.Imagenes
import java.text.SimpleDateFormat

/** Vista previa de una foto de la galeria (p. 20) antes de analizarla. */
class VistaPreviaActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_URI = "uri"

        fun abrir(context: Context, uri: Uri) =
            context.startActivity(Intent(context, VistaPreviaActivity::class.java).putExtra(EXTRA_URI, uri.toString()))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityVistaPreviaBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.barra.setNavigationOnClickListener { finish() }

        val uri = Uri.parse(intent.getStringExtra(EXTRA_URI) ?: return finish())
        b.botonAnalizar.setOnClickListener { AnalisisActivity.abrir(this, uri); finish() }
        b.botonOtra.setOnClickListener { finish(); MainActivity.volver(this, MainActivity.ACCION_GALERIA) }

        lifecycleScope.launch {
            val foto = withContext(Dispatchers.IO) { Imagenes.cargar(this@VistaPreviaActivity, uri) }
            b.imagen.setImageBitmap(foto)
            val (nombre, datos) = withContext(Dispatchers.IO) { describir(uri) }
            b.textoArchivo.text = nombre
            b.textoDatos.text = datos
        }
    }

    /** Nombre del archivo y "fecha · ancho × alto" (de EXIF si existe). */
    private fun describir(uri: Uri): Pair<String, String> {
        val nombre = contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use {
            if (it.moveToFirst()) it.getString(0) else null
        } ?: uri.lastPathSegment.orEmpty()
        val partes = mutableListOf<String>()
        runCatching {
            contentResolver.openInputStream(uri)?.use { s ->
                val exif = ExifInterface(s)
                exif.getAttribute(ExifInterface.TAG_DATETIME_ORIGINAL)?.let { t ->
                    SimpleDateFormat("yyyy:MM:dd HH:mm:ss", Textos.LOCALE).parse(t)?.let { partes += Textos.fechaHora(it.time) }
                }
                val w = exif.getAttributeInt(ExifInterface.TAG_IMAGE_WIDTH, 0)
                val h = exif.getAttributeInt(ExifInterface.TAG_IMAGE_LENGTH, 0)
                if (w > 0 && h > 0) partes += "$w × $h"
            }
        }
        return nombre to partes.joinToString(" · ")
    }
}
