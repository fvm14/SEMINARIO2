package pe.edu.ulima.paltascan.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityCamaraVivoBinding
import pe.edu.ulima.paltascan.datos.Imagenes
import pe.edu.ulima.paltascan.ml.ResultadoAnalisis
import pe.edu.ulima.paltascan.ml.ValidacionCaptura.Problema
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

/**
 * Camara en vivo: analiza continuamente el cuadro mas reciente con el modelo
 * elegido y muestra el resultado en el recuadro de arriba a la derecha. Los
 * cuadros que llegan mientras el modelo trabaja se descartan, asi que la
 * frecuencia de actualizacion (FPS) la fija la latencia del analisis.
 * No guarda nada en el historial.
 */
class CamaraVivoActivity : AppCompatActivity() {

    private lateinit var b: ActivityCamaraVivoBinding
    private lateinit var ejecutor: ExecutorService

    /** Instantes de los ultimos resultados, para el FPS de una ventana movil. */
    private val marcas = ArrayDeque<Long>()

    private val pedirPermiso = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) iniciarCamara() else {
            Toast.makeText(this, R.string.vivo_permiso, Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityCamaraVivoBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        ejecutor = Executors.newSingleThreadExecutor()
        mostrarIndicacion(if (app.usarGpu) "GPU" else "CPU")

        val permitido = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        if (permitido) iniciarCamara() else pedirPermiso.launch(Manifest.permission.CAMERA)
    }

    override fun onDestroy() {
        super.onDestroy()
        ejecutor.shutdown()
    }

    private fun iniciarCamara() {
        val futuro = ProcessCameraProvider.getInstance(this)
        futuro.addListener({
            // 4:3 como las fotos del dataset; el analisis se hace a ~1280x960 y luego se reduce a 800 px (letterbox).
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build()
            val vista = Preview.Builder().setResolutionSelector(selector).build()
            vista.setSurfaceProvider(b.vistaCamara.surfaceProvider)
            val analisis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
            analisis.setAnalyzer(ejecutor, ::analizarCuadro)
            val proveedor = futuro.get()
            proveedor.unbindAll()
            proveedor.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, vista, analisis)
        }, ContextCompat.getMainExecutor(this))
    }

    /** Corre en el hilo del ejecutor: un cuadro a la vez. */
    private fun analizarCuadro(imagen: ImageProxy) {
        val foto = try {
            girar(imagen.toBitmap(), imagen.imageInfo.rotationDegrees)
        } finally {
            imagen.close()
        }
        val analizador = app.analizador ?: return
        val tamano = "${foto.width}x${foto.height}"
        val r = try {
            analizador.analizar(foto)
        } catch (e: Exception) {
            Log.e(TAG, "Error al analizar el cuadro", e)
            return
        } finally {
            foto.recycle()
        }
        val capa = Imagenes.contornos(r)
        val ahora = SystemClock.elapsedRealtime()
        Log.i(TAG, "${analizador.modelo.id} ${analizador.procesador} cuadro $tamano pre ${r.tiempos.preprocesoMs.roundToInt()} " +
            "inf ${r.tiempos.inferenciaMs.roundToInt()} post ${r.tiempos.postprocesoMs.roundToInt()} ms")
        runOnUiThread {
            if (isDestroyed) return@runOnUiThread
            b.capaContornos.setImageBitmap(capa)
            mostrarIndicacion(analizador.procesador)
            mostrar(r, ahora)
        }
    }

    private fun mostrarIndicacion(procesador: String) {
        b.textoIndicacion.text = getString(R.string.vivo_indicacion) + "\n" + app.modeloActual.nombre + " · " + procesador
    }

    private fun mostrar(r: ResultadoAnalisis, ahora: Long) {
        marcas.addLast(ahora)
        while (marcas.size > 6) marcas.removeFirst()
        val fps = if (marcas.size > 1) (marcas.size - 1) * 1000.0 / (marcas.last() - marcas.first()) else 0.0
        b.textoFps.text = getString(R.string.vivo_fps, fps, r.tiempos.totalMs.roundToInt())

        val bloqueante = r.bloqueante
        if (bloqueante != null) {
            b.textoCategoria.text = getString(
                when (bloqueante) {
                    Problema.VARIAS_PALTAS -> R.string.vivo_varias
                    Problema.PALTA_CORTADA -> R.string.vivo_cortada
                    else -> R.string.vivo_sin_palta
                }
            )
            b.textoCategoria.setBackgroundColor(0xFF424242.toInt())
            b.textoDefecto.text = "—"
            b.textoMadurez.text = ""
            b.textoConfianza.text = ""
            b.textoAviso.visibility = View.GONE
            return
        }

        b.textoCategoria.text = Textos.CATEGORIAS[r.categoria]
        b.textoCategoria.setBackgroundColor(ContextCompat.getColor(this, Textos.colorCategoria(r.categoria)))
        b.textoDefecto.text = getString(R.string.vivo_defecto, Textos.porcentaje(r.ratio))
        b.textoMadurez.text = getString(R.string.vivo_madurez, r.madurez, Textos.nombreMadurez(r.madurez))
        val confianza = (r.probMadurez * 100).roundToInt()
        val dudosa = Problema.BAJA_CONFIANZA in r.avisos
        b.textoConfianza.text = getString(if (dudosa) R.string.vivo_dudosa else R.string.vivo_confianza, confianza)
        val avisos = r.avisos.mapNotNull {
            when (it) {
                Problema.OSCURA -> getString(R.string.aviso_oscura)
                Problema.BORROSA -> getString(R.string.aviso_borrosa)
                else -> null
            }
        }
        b.textoAviso.text = avisos.joinToString(" · ")
        b.textoAviso.visibility = if (avisos.isEmpty()) View.GONE else View.VISIBLE
    }

    private fun girar(foto: Bitmap, grados: Int): Bitmap {
        if (grados == 0) return foto
        val m = Matrix().apply { postRotate(grados.toFloat()) }
        return Bitmap.createBitmap(foto, 0, 0, foto.width, foto.height, m, true).also { if (it !== foto) foto.recycle() }
    }

    companion object {
        private const val TAG = "PaltaScanVivo"
    }
}
