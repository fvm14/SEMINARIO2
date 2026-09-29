package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityAnalisisBinding
import pe.edu.ulima.paltascan.datos.Imagenes
import pe.edu.ulima.paltascan.datos.Inspeccion

/**
 * "Analizando..." (p. 10): corre el modelo en el celular y decide a donde ir.
 * Si la foto no sirve (sin palta, varias o cortada) abre la pantalla de error
 * sin guardar nada; si sirve, guarda el analisis en el historial y abre el
 * resultado.
 */
class AnalisisActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_URI = "uri"

        fun abrir(context: Context, uri: Uri) =
            context.startActivity(Intent(context, AnalisisActivity::class.java).putExtra(EXTRA_URI, uri.toString()))
    }

    private val sinVolver = object : OnBackPressedCallback(true) {
        override fun handleOnBackPressed() = Unit // no se interrumpe el analisis
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityAnalisisBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        onBackPressedDispatcher.addCallback(this, sinVolver)

        val uri = Uri.parse(intent.getStringExtra(EXTRA_URI) ?: return finish())
        val analizador = app.analizador ?: return finish()

        lifecycleScope.launch {
            try {
                val foto = withContext(Dispatchers.IO) { Imagenes.cargar(this@AnalisisActivity, uri) }
                b.imagenFoto.setImageBitmap(foto)
                val r = withContext(Dispatchers.Default) { analizador.analizar(foto) }
                val bloqueante = r.bloqueante
                if (bloqueante != null) {
                    ErrorAnalisisActivity.abrir(this@AnalisisActivity, bloqueante)
                } else {
                    val id = withContext(Dispatchers.IO) {
                        val marca = System.currentTimeMillis()
                        val rutaFoto = if (app.guardarFotos) Imagenes.guardar(this@AnalisisActivity, foto, "$marca.jpg") else ""
                        val rutaOverlay = Imagenes.guardar(this@AnalisisActivity, Imagenes.superponer(foto, r), "${marca}_analisis.jpg")
                        app.baseDatos.insertar(
                            Inspeccion(
                                loteId = null, fechaMs = marca,
                                archivoFoto = rutaFoto, archivoOverlay = rutaOverlay,
                                madurez = r.madurez, probMadurez = r.probMadurez,
                                ratio = r.ratio, categoria = r.categoria,
                                msPreproceso = r.tiempos.preprocesoMs,
                                msInferencia = r.tiempos.inferenciaMs,
                                msPostproceso = r.tiempos.postprocesoMs,
                                avisos = r.avisos.joinToString(",") { it.name },
                            )
                        )
                    }
                    ResultadoActivity.abrirNuevo(this@AnalisisActivity, id)
                }
            } catch (e: Exception) {
                Toast.makeText(this@AnalisisActivity, getString(R.string.error_analisis, e.message), Toast.LENGTH_LONG).show()
            } finally {
                finish()
            }
        }
    }
}
