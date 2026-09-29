package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityResultadoBinding
import pe.edu.ulima.paltascan.ml.Ocde

class ResultadoActivity : AppCompatActivity() {

    companion object {
        const val EXTRA_ID = "id_inspeccion"

        fun colorCategoria(categoria: Int) = when (categoria) {
            0 -> R.color.cat_i
            1 -> R.color.cat_ii
            else -> R.color.rechazo
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityResultadoBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        val i = app.baseDatos.obtener(intent.getLongExtra(EXTRA_ID, -1)) ?: return finish()

        b.tarjetaCategoria.setCardBackgroundColor(ContextCompat.getColor(this, colorCategoria(i.categoria)))
        b.textoCategoria.text = Ocde.CATEGORIAS[i.categoria]
        b.imagenOriginal.setImageBitmap(BitmapFactory.decodeFile(i.archivoFoto))
        b.imagenAnalisis.setImageBitmap(BitmapFactory.decodeFile(i.archivoOverlay))
        b.textoMadurez.text = getString(R.string.madurez_valor, i.madurez, i.probMadurez * 100)
        b.textoRatio.text = getString(R.string.ratio_valor, i.ratio * 100)
        b.textoTiempos.text = getString(R.string.tiempos_valor, i.msTotal, i.msPreproceso, i.msInferencia, i.msPostproceso)

        b.botonOtra.setOnClickListener { finish() }
        b.botonHistorial.setOnClickListener {
            startActivity(Intent(this, HistorialActivity::class.java))
            finish()
        }
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }
}
