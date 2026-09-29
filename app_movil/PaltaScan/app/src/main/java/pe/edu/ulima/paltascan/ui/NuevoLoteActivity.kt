package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityNuevoLoteBinding
import pe.edu.ulima.paltascan.datos.Lote

/** Nuevo lote (p. 27). Con EXTRA_DEVOLVER devuelve el id creado en vez de abrir el lote. */
class NuevoLoteActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_DEVOLVER = "devolver"
        const val EXTRA_LOTE_ID = "lote_id"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityNuevoLoteBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.barra.setNavigationOnClickListener { finish() }

        val ahora = System.currentTimeMillis()
        b.textoFecha.text = getString(R.string.fecha_valor, Textos.fecha(ahora))

        b.botonGuardar.setOnClickListener {
            val nombre = b.editNombre.text?.toString()?.trim().orEmpty()
            if (nombre.isEmpty()) {
                b.campoNombre.error = getString(R.string.falta_nombre)
                return@setOnClickListener
            }
            val id = app.baseDatos.crearLote(
                Lote(
                    nombre = nombre,
                    productor = b.editProductor.text?.toString()?.trim().orEmpty(),
                    notas = b.editNotas.text?.toString()?.trim().orEmpty(),
                    fechaMs = ahora,
                )
            )
            if (intent.getBooleanExtra(EXTRA_DEVOLVER, false)) {
                setResult(RESULT_OK, Intent().putExtra(EXTRA_LOTE_ID, id))
            } else {
                LoteActivity.abrir(this, id)
            }
            finish()
        }
    }
}
