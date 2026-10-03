package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityMainBinding
import pe.edu.ulima.paltascan.datos.Imagenes
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.ml.ValidacionCaptura.Problema
import java.io.File

/**
 * Pantalla principal: lote actual, captura (camara o galeria) y resultado del
 * ultimo analisis en la misma pantalla. Cada analisis valido se guarda en el
 * historial, dentro del lote actual.
 */
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
        uriCaptura = savedInstanceState?.getString("uri")?.let(Uri::parse)

        b.botonCamara.setOnClickListener { abrirCamara() }
        b.botonGaleria.setOnClickListener {
            elegirFoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        b.botonVivo.setOnClickListener { startActivity(Intent(this, CamaraVivoActivity::class.java)) }
        b.botonHistorial.setOnClickListener { startActivity(Intent(this, HistorialActivity::class.java)) }
        b.botonLote.setOnClickListener {
            Acciones.elegirLote(this) { id -> app.loteActualId = id; mostrarLote() }
        }
        b.botonModelo.setOnClickListener { elegirModelo() }
        b.botonProcesador.setOnClickListener { elegirProcesador() }
        cargarModelo()
    }

    override fun onResume() {
        super.onResume()
        mostrarLote()
        mostrarModelo()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        uriCaptura?.let { outState.putString("uri", it.toString()) }
    }

    private fun mostrarLote() {
        b.textoLote.text = Acciones.nombreLote(this, app.loteActualId)
    }

    private fun mostrarModelo() {
        b.filaModelo.visibility = if (app.modelosDisponibles.size > 1) View.VISIBLE else View.GONE
        b.textoModeloActual.text = app.modeloActual.nombre
        b.filaProcesador.visibility = if (app.gpuDisponible) View.VISIBLE else View.GONE
        b.textoProcesador.setText(if (app.usarGpu) R.string.procesador_gpu else R.string.procesador_cpu)
    }

    /** CPU o GPU (delegado de LiteRT), para comparar la latencia. */
    private fun elegirProcesador() {
        val opciones = arrayOf(getString(R.string.procesador_cpu), getString(R.string.procesador_gpu))
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.elegir_procesador)
            .setSingleChoiceItems(opciones, if (app.usarGpu) 1 else 0) { d, i ->
                d.dismiss()
                if ((i == 1) != app.usarGpu) {
                    app.usarGpu = i == 1
                    mostrarModelo()
                    cargarModelo()
                }
            }
            .show()
    }

    /** Solo para la comparacion de modelos: el resultado guardado indica con cual se hizo. */
    private fun elegirModelo() {
        val modelos = app.modelosDisponibles
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.elegir_modelo)
            .setSingleChoiceItems(modelos.map { it.nombre }.toTypedArray(), modelos.indexOf(app.modeloActual)) { d, i ->
                d.dismiss()
                if (modelos[i] != app.modeloActual) {
                    app.modeloActual = modelos[i]
                    mostrarModelo()
                    b.resultado.root.visibility = View.GONE
                    b.textoMensaje.visibility = View.GONE
                    cargarModelo()
                }
            }
            .show()
    }

    private fun cargarModelo() {
        habilitar(false)
        b.textoEstado.setText(R.string.cargando_modelo)
        lifecycleScope.launch {
            val analizador = withContext(Dispatchers.Default) { app.analizador }
            b.textoEstado.setText(if (analizador != null) R.string.subtitulo else R.string.modelo_faltante)
            if (analizador != null && analizador.pidioGpu && !analizador.gpu) {
                Toast.makeText(this@MainActivity, R.string.gpu_fallo, Toast.LENGTH_LONG).show()
            }
            habilitar(analizador != null)
        }
    }

    private fun abrirCamara() {
        val dir = File(cacheDir, "capturas").apply { mkdirs() }
        val archivo = File(dir, "captura_${System.currentTimeMillis()}.jpg")
        dir.listFiles()?.filter { it != archivo }?.forEach { it.delete() }
        val uri = FileProvider.getUriForFile(this, "$packageName.fotos", archivo)
        uriCaptura = uri
        tomarFoto.launch(uri)
    }

    private fun analizar(uri: Uri) {
        val analizador = app.analizador ?: return
        habilitar(false)
        b.panelProgreso.visibility = View.VISIBLE
        b.textoMensaje.visibility = View.GONE
        b.resultado.root.visibility = View.GONE
        lifecycleScope.launch {
            try {
                val foto = withContext(Dispatchers.IO) { Imagenes.cargar(this@MainActivity, uri) }
                val r = withContext(Dispatchers.Default) { analizador.analizar(foto) }
                val bloqueante = r.bloqueante
                if (bloqueante != null) {
                    mostrarMensaje(getString(R.string.no_analizable) + ". " + getString(motivo(bloqueante)), error = true)
                    return@launch
                }
                val loteId = app.loteActualId
                val id = withContext(Dispatchers.IO) {
                    val marca = System.currentTimeMillis()
                    app.baseDatos.insertar(
                        Inspeccion(
                            loteId = loteId, fechaMs = marca,
                            archivoFoto = Imagenes.guardar(this@MainActivity, foto, "$marca.jpg"),
                            archivoOverlay = Imagenes.guardar(this@MainActivity, Imagenes.superponer(foto, r), "${marca}_analisis.jpg"),
                            madurez = r.madurez, probMadurez = r.probMadurez,
                            ratio = r.ratio, categoria = r.categoria,
                            msPreproceso = r.tiempos.preprocesoMs,
                            msInferencia = r.tiempos.inferenciaMs,
                            msPostproceso = r.tiempos.postprocesoMs,
                            avisos = r.avisos.joinToString(",") { it.name },
                            modelo = analizador.modelo.id,
                        )
                    )
                }
                app.baseDatos.obtener(id)?.let {
                    b.resultado.mostrar(it)
                    b.resultado.root.visibility = View.VISIBLE
                    mostrarMensaje(getString(R.string.guardado_en, Acciones.nombreLote(this@MainActivity, loteId)), error = false)
                }
            } catch (e: Exception) {
                Toast.makeText(this@MainActivity, getString(R.string.error_analisis, e.message), Toast.LENGTH_LONG).show()
                b.textoMensaje.visibility = View.VISIBLE
            } finally {
                b.panelProgreso.visibility = View.GONE
                habilitar(true)
            }
        }
    }

    private fun motivo(p: Problema) = when (p) {
        Problema.VARIAS_PALTAS -> R.string.motivo_varias
        Problema.PALTA_CORTADA -> R.string.motivo_cortada
        else -> R.string.motivo_sin_palta
    }

    private fun mostrarMensaje(texto: String, error: Boolean) {
        b.textoMensaje.text = texto
        b.textoMensaje.setTextColor(ContextCompat.getColor(this, if (error) R.color.rechazo else R.color.texto_suave))
        b.textoMensaje.visibility = View.VISIBLE
    }

    private fun habilitar(si: Boolean) {
        b.botonCamara.isEnabled = si
        b.botonGaleria.isEnabled = si
        b.botonVivo.isEnabled = si
        b.botonModelo.isEnabled = si
        b.botonProcesador.isEnabled = si
    }
}
