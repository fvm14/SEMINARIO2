package pe.edu.ulima.paltascan.ui

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.AdapterView
import android.widget.ArrayAdapter
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.DividerItemDecoration
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityHistorialBinding
import pe.edu.ulima.paltascan.databinding.ItemInspeccionBinding
import pe.edu.ulima.paltascan.databinding.VistaResultadoBinding
import pe.edu.ulima.paltascan.datos.ExportadorCsv
import pe.edu.ulima.paltascan.datos.ExportadorPdf
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.datos.Lote
import pe.edu.ulima.paltascan.datos.Resumen

/**
 * Historial: todos los analisis o los de un lote, con su resumen por
 * categoria y madurez y la exportacion a CSV o PDF. Tocar un analisis abre
 * su detalle (mover a otro lote o eliminar).
 */
class HistorialActivity : AppCompatActivity() {

    private lateinit var b: ActivityHistorialBinding
    private var lotes: List<Lote> = emptyList()
    private var inspecciones: List<Inspeccion> = emptyList()

    /** null = todos los analisis */
    private val loteElegido: Lote? get() = b.selectorLote.selectedItemPosition.let { if (it <= 0) null else lotes[it - 1] }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityHistorialBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.botonAtras.setOnClickListener { finish() }
        b.lista.layoutManager = LinearLayoutManager(this)
        b.lista.addItemDecoration(DividerItemDecoration(this, DividerItemDecoration.VERTICAL))
        b.selectorLote.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) = cargar()
            override fun onNothingSelected(p: AdapterView<*>?) = Unit
        }
        b.botonCsv.setOnClickListener { exportar("csv") }
        b.botonPdf.setOnClickListener { exportar("pdf") }
    }

    override fun onResume() {
        super.onResume()
        val previo = loteElegido?.id ?: app.loteActualId
        lotes = app.baseDatos.lotes()
        val nombres = listOf(getString(R.string.todos)) + lotes.map { it.nombre }
        b.selectorLote.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, nombres)
        b.selectorLote.setSelection(lotes.indexOfFirst { it.id == previo } + 1)
        cargar()
    }

    private fun cargar() {
        val lote = loteElegido
        inspecciones = if (lote == null) app.baseDatos.historial() else app.baseDatos.porLote(lote.id)
        val r = Resumen(inspecciones)
        b.textoResumen.text = getString(R.string.resumen, r.total, r.porCategoria[0], r.porCategoria[1], r.porCategoria[2])
        b.textoMadurez.text = getString(R.string.resumen_madurez, r.porMadurez.joinToString(" · "))
        b.textoVacio.visibility = if (inspecciones.isEmpty()) View.VISIBLE else View.GONE
        b.botonCsv.isEnabled = inspecciones.isNotEmpty()
        b.botonPdf.isEnabled = inspecciones.isNotEmpty() && lote != null
        b.lista.adapter = Adaptador(inspecciones)
    }

    private fun exportar(tipo: String) {
        val lote = loteElegido
        lifecycleScope.launch {
            val archivo = Acciones.archivoExporte(this@HistorialActivity, ExportadorCsv.nombreArchivo(lote, tipo))
            withContext(Dispatchers.IO) {
                if (tipo == "pdf" && lote != null) ExportadorPdf.generar(lote, inspecciones, archivo)
                else archivo.writeText(ExportadorCsv.generar(lote, inspecciones))
            }
            Acciones.compartirArchivo(this@HistorialActivity, archivo, if (tipo == "pdf") "application/pdf" else "text/csv")
        }
    }

    private fun detalle(i: Inspeccion) {
        val v = VistaResultadoBinding.inflate(layoutInflater)
        v.mostrar(i)
        val dp = resources.displayMetrics.density
        v.root.setPadding((20 * dp).toInt(), (8 * dp).toInt(), (20 * dp).toInt(), 0)
        MaterialAlertDialogBuilder(this)
            .setTitle(Textos.fechaHora(i.fechaMs) + " · " + Acciones.nombreLote(this, i.loteId))
            .setView(android.widget.ScrollView(this).apply { addView(v.root) })
            .setPositiveButton(R.string.cerrar, null)
            .setNeutralButton(R.string.mover_a_lote) { _, _ ->
                Acciones.elegirLote(this) { id -> app.baseDatos.moverALote(i.id, id); onResume() }
            }
            .setNegativeButton(R.string.eliminar) { _, _ -> confirmarEliminar(i) }
            .show()
    }

    private fun confirmarEliminar(i: Inspeccion) {
        MaterialAlertDialogBuilder(this)
            .setTitle(R.string.eliminar_titulo)
            .setMessage(R.string.eliminar_texto)
            .setNegativeButton(R.string.cancelar, null)
            .setPositiveButton(R.string.eliminar) { _, _ -> Acciones.eliminar(this, i); cargar() }
            .show()
    }

    private inner class Adaptador(val items: List<Inspeccion>) : RecyclerView.Adapter<Adaptador.Fila>() {
        inner class Fila(val v: ItemInspeccionBinding) : RecyclerView.ViewHolder(v.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Fila(ItemInspeccionBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(f: Fila, pos: Int) {
            val i = items[pos]
            f.v.textoCategoria.text = Textos.CATEGORIAS[i.categoria]
            f.v.textoCategoria.setTextColor(ContextCompat.getColor(this@HistorialActivity, Textos.colorCategoria(i.categoria)))
            f.v.textoDetalle.text = "%s · madurez %d · defecto %s".format(
                Textos.fechaHora(i.fechaMs), i.madurez, Textos.porcentaje(i.ratio)
            )
            f.v.miniatura.setImageBitmap(BitmapFactory.decodeFile(i.archivoOverlay, BitmapFactory.Options().apply { inSampleSize = 8 }))
            f.v.root.setOnClickListener { detalle(i) }
        }
    }
}
