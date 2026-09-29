package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityHistorialBinding
import pe.edu.ulima.paltascan.databinding.ItemInspeccionBinding
import pe.edu.ulima.paltascan.datos.ExportadorCsv
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.ml.Ocde
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistorialActivity : AppCompatActivity() {

    private lateinit var b: ActivityHistorialBinding
    private var inspecciones: List<Inspeccion> = emptyList()

    private val crearCsv = registerForActivityResult(ActivityResultContracts.CreateDocument("text/csv")) { uri ->
        uri ?: return@registerForActivityResult
        contentResolver.openOutputStream(uri)?.use { it.write(ExportadorCsv.generar(inspecciones).toByteArray()) }
        Toast.makeText(this, R.string.exportado, Toast.LENGTH_SHORT).show()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityHistorialBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        b.lista.layoutManager = LinearLayoutManager(this)
        b.botonExportar.setOnClickListener {
            crearCsv.launch(app.loteActual.replace(Regex("[^A-Za-z0-9-]+"), "_") + ".csv")
        }
    }

    override fun onResume() {
        super.onResume()
        val lote = app.loteActual
        inspecciones = app.baseDatos.porLote(lote)
        b.textoLote.text = lote
        val porCategoria = IntArray(3).also { c -> inspecciones.forEach { c[it.categoria]++ } }
        b.textoResumen.text = getString(R.string.resumen_lote, inspecciones.size, porCategoria[0], porCategoria[1], porCategoria[2])
        val porMadurez = IntArray(5).also { c -> inspecciones.forEach { c[it.madurez - 1]++ } }
        b.textoMadurez.text = "Madurez 1-5: " + porMadurez.joinToString(" · ")
        b.textoVacio.visibility = if (inspecciones.isEmpty()) View.VISIBLE else View.GONE
        b.botonExportar.isEnabled = inspecciones.isNotEmpty()
        b.lista.adapter = Adaptador(inspecciones)
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    private inner class Adaptador(val items: List<Inspeccion>) : RecyclerView.Adapter<Adaptador.Fila>() {
        private val hora = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

        inner class Fila(val v: ItemInspeccionBinding) : RecyclerView.ViewHolder(v.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Fila(ItemInspeccionBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(f: Fila, pos: Int) {
            val i = items[pos]
            f.v.textoCategoria.text = Ocde.CATEGORIAS[i.categoria]
            f.v.textoCategoria.setTextColor(ContextCompat.getColor(this@HistorialActivity, ResultadoActivity.colorCategoria(i.categoria)))
            f.v.textoDetalle.text = "%s · madurez %d · defecto %.1f%% · %.0f ms".format(
                hora.format(Date(i.fechaMs)), i.madurez, i.ratio * 100, i.msTotal
            )
            f.v.miniatura.setImageBitmap(BitmapFactory.decodeFile(i.archivoOverlay, BitmapFactory.Options().apply { inSampleSize = 8 }))
            f.v.root.setOnClickListener {
                startActivity(Intent(this@HistorialActivity, ResultadoActivity::class.java).putExtra(ResultadoActivity.EXTRA_ID, i.id))
            }
        }
    }
}
