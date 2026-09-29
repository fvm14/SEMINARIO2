package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.datos.Imagenes
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.databinding.ActivityMainBinding
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var b: ActivityMainBinding
    private var uriCaptura: Uri? = null

    private val tomarFoto = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) uriCaptura?.let(::analizar)
    }

    private val elegirFoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let(::analizar)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()

        b.botonNuevoLote.setOnClickListener {
            app.nuevoLote()
            mostrarLote()
        }
        b.botonCamara.setOnClickListener { abrirCamara() }
        b.botonGaleria.setOnClickListener {
            elegirFoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        b.botonHistorial.setOnClickListener { startActivity(Intent(this, HistorialActivity::class.java)) }

        verificarModelo()
    }

    override fun onResume() {
        super.onResume()
        mostrarLote()
    }

    private fun mostrarLote() {
        b.textoLote.text = app.loteActual
    }

    private fun verificarModelo() {
        lifecycleScope.launch {
            val listo = withContext(Dispatchers.Default) { app.analizador != null }
            b.textoModelo.setText(if (listo) R.string.modelo_listo else R.string.modelo_faltante)
            b.botonCamara.isEnabled = listo
            b.botonGaleria.isEnabled = listo
        }
    }

    private fun abrirCamara() {
        val dir = File(cacheDir, "capturas").apply { mkdirs() }
        val archivo = File(dir, "captura.jpg")
        val uri = FileProvider.getUriForFile(this, "$packageName.fotos", archivo)
        uriCaptura = uri
        tomarFoto.launch(uri)
    }

    private fun analizar(uri: Uri) {
        val analizador = app.analizador ?: return
        ocupado(true)
        lifecycleScope.launch {
            try {
                val id = withContext(Dispatchers.Default) {
                    val foto = Imagenes.cargar(this@MainActivity, uri)
                    val r = analizador.analizar(foto)
                    if (!r.frutoDetectado) return@withContext null
                    val marca = System.currentTimeMillis()
                    val rutaFoto = Imagenes.guardar(this@MainActivity, foto, "$marca.jpg")
                    val rutaOverlay = Imagenes.guardar(this@MainActivity, Imagenes.superponer(foto, r), "${marca}_analisis.jpg")
                    app.baseDatos.insertar(
                        Inspeccion(
                            lote = app.loteActual, fechaMs = marca,
                            archivoFoto = rutaFoto, archivoOverlay = rutaOverlay,
                            madurez = r.madurez, probMadurez = r.probMadurez,
                            ratio = r.ratio, categoria = r.categoria,
                            msPreproceso = r.tiempos.preprocesoMs,
                            msInferencia = r.tiempos.inferenciaMs,
                            msPostproceso = r.tiempos.postprocesoMs,
                        )
                    )
                }
                if (id == null) {
                    Toast.makeText(this@MainActivity, R.string.sin_fruto, Toast.LENGTH_LONG).show()
                } else {
                    startActivity(Intent(this@MainActivity, ResultadoActivity::class.java).putExtra(ResultadoActivity.EXTRA_ID, id))
                }
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, getString(R.string.error_analisis, e.message), Toast.LENGTH_LONG).show()
            } finally {
                ocupado(false)
            }
        }
    }

    private fun ocupado(si: Boolean) {
        b.panelProgreso.visibility = if (si) View.VISIBLE else View.GONE
        b.botonCamara.isEnabled = !si
        b.botonGaleria.isEnabled = !si
    }
}
