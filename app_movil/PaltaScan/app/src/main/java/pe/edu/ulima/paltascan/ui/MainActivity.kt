package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.databinding.ActivityMainBinding
import java.io.File

/**
 * Contenedor con la barra inferior (Inicio, Historial, Lotes, Ajustes).
 * Tambien centraliza la captura: camara del sistema o selector de galeria.
 */
class MainActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_ACCION = "accion"
        const val ACCION_CAMARA = "camara"
        const val ACCION_GALERIA = "galeria"
        const val ACCION_HISTORIAL = "historial"

        /** Vuelve al inicio (cerrando lo que haya encima) y, si se pide, abre la camara o la galeria. */
        fun volver(context: Context, accion: String? = null) {
            context.startActivity(Intent(context, MainActivity::class.java).apply {
                addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP)
                accion?.let { putExtra(EXTRA_ACCION, it) }
            })
        }
    }

    private lateinit var b: ActivityMainBinding
    private var uriCaptura: Uri? = null

    private val tomarFoto = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        if (ok) uriCaptura?.let { AnalisisActivity.abrir(this, it) }
    }

    private val elegirFoto = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        uri?.let { VistaPreviaActivity.abrir(this, it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityMainBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        uriCaptura = savedInstanceState?.getString("uri")?.let(Uri::parse)

        b.navegacion.setOnItemSelectedListener { item ->
            mostrar(item.itemId)
            true
        }
        if (savedInstanceState == null) b.navegacion.selectedItemId = R.id.nav_inicio
        atenderAccion(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        atenderAccion(intent)
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        uriCaptura?.let { outState.putString("uri", it.toString()) }
    }

    private fun atenderAccion(intent: Intent) {
        when (intent.getStringExtra(EXTRA_ACCION)) {
            ACCION_CAMARA -> abrirCamara()
            ACCION_GALERIA -> abrirGaleria()
            ACCION_HISTORIAL -> irA(R.id.nav_historial)
        }
        intent.removeExtra(EXTRA_ACCION)
    }

    fun irA(itemId: Int) {
        b.navegacion.selectedItemId = itemId
    }

    private fun mostrar(itemId: Int) {
        val fragment: Fragment = when (itemId) {
            R.id.nav_historial -> HistorialFragment()
            R.id.nav_lotes -> LotesFragment()
            R.id.nav_ajustes -> AjustesFragment()
            else -> InicioFragment()
        }
        supportFragmentManager.beginTransaction().replace(R.id.contenedor, fragment).commit()
    }

    fun abrirCamara() {
        val dir = File(cacheDir, "capturas").apply { mkdirs() }
        val archivo = File(dir, "captura_${System.currentTimeMillis()}.jpg")
        dir.listFiles()?.filter { it != archivo }?.forEach { it.delete() }
        val uri = FileProvider.getUriForFile(this, "$packageName.fotos", archivo)
        uriCaptura = uri
        tomarFoto.launch(uri)
    }

    fun abrirGaleria() {
        elegirFoto.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
    }
}
