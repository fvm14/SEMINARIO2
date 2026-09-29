package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.bottomsheet.BottomSheetDialog
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityLoteBinding
import pe.edu.ulima.paltascan.databinding.CabeceraLoteBinding
import pe.edu.ulima.paltascan.databinding.HojaExportarBinding
import pe.edu.ulima.paltascan.datos.ExportadorCsv
import pe.edu.ulima.paltascan.datos.ExportadorPdf
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.datos.Lote
import pe.edu.ulima.paltascan.datos.Resumen

/** Detalle de un lote (p. 28-29): resumen por categoria y madurez, analisis y exportacion. */
class LoteActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_ID = "lote_id"

        fun abrir(context: Context, id: Long) =
            context.startActivity(Intent(context, LoteActivity::class.java).putExtra(EXTRA_ID, id))
    }

    private lateinit var b: ActivityLoteBinding
    private lateinit var lote: Lote
    private var inspecciones: List<Inspeccion> = emptyList()
    private var cabecera: CabeceraLoteBinding? = null

    private val adaptador = AdaptadorInspecciones { parent ->
        val c = CabeceraLoteBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        cabecera = c
        llenarCabecera(c)
        object : RecyclerView.ViewHolder(c.root) {}
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityLoteBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.barra.setNavigationOnClickListener { finish() }
        b.lista.layoutManager = LinearLayoutManager(this)
        b.lista.adapter = adaptador
        b.botonExportar.setOnClickListener { mostrarExportar() }
    }

    override fun onResume() {
        super.onResume()
        lote = app.baseDatos.lote(intent.getLongExtra(EXTRA_ID, -1)) ?: return finish()
        inspecciones = app.baseDatos.porLote(lote.id)
        b.barra.title = lote.nombre
        b.botonExportar.isEnabled = inspecciones.isNotEmpty()
        cabecera?.let(::llenarCabecera)
        adaptador.mostrar(this, inspecciones, agruparPorDia = false)
    }

    private fun llenarCabecera(c: CabeceraLoteBinding) {
        val origen = listOf(lote.productor, Textos.fecha(lote.fechaMs)).filter { it.isNotBlank() }
        c.textoOrigen.text = origen.joinToString(" · ")
        c.textoTotal.text = resources.getQuantityString(R.plurals.n_paltas_analizadas, inspecciones.size, inspecciones.size)
        c.resumen.mostrar(Resumen(inspecciones))
        c.textoNotas.text = lote.notas
        c.textoNotas.visibility = if (lote.notas.isBlank()) View.GONE else View.VISIBLE
        c.textoTituloLista.text = getString(R.string.analisis_del_lote) + "  " + inspecciones.size
        c.textoLoteVacio.visibility = if (inspecciones.isEmpty()) View.VISIBLE else View.GONE
    }

    private fun mostrarExportar() {
        val hoja = BottomSheetDialog(this)
        val h = HojaExportarBinding.inflate(layoutInflater)
        h.textoLote.text = lote.nombre + " · " + resources.getQuantityString(R.plurals.n_paltas, inspecciones.size, inspecciones.size)
        h.opcionCsv.setOnClickListener { hoja.dismiss(); exportarCsv() }
        h.opcionPdf.setOnClickListener { hoja.dismiss(); exportarPdf() }
        hoja.setContentView(h.root)
        hoja.show()
    }

    private fun exportarPdf() {
        lifecycleScope.launch {
            val archivo = Acciones.archivoExporte(this@LoteActivity, ExportadorCsv.nombreArchivo(lote, "pdf"))
            withContext(Dispatchers.IO) { ExportadorPdf.generar(lote, inspecciones, archivo) }
            Acciones.compartirArchivo(this@LoteActivity, archivo, "application/pdf")
        }
    }

    private fun exportarCsv() {
        val archivo = Acciones.archivoExporte(this, ExportadorCsv.nombreArchivo(lote, "csv"))
        archivo.writeText(ExportadorCsv.generar(lote, inspecciones))
        Acciones.compartirArchivo(this, archivo, "text/csv")
        Toast.makeText(this, archivo.name, Toast.LENGTH_SHORT).show()
    }
}
