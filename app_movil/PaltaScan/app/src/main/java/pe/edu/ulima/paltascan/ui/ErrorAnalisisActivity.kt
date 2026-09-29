package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.databinding.ActivityErrorBinding
import pe.edu.ulima.paltascan.ml.ValidacionCaptura.Problema

/** "No pudimos analizar la foto" (p. 15), con el motivo detectado. */
class ErrorAnalisisActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_MOTIVO = "motivo"

        fun abrir(context: Context, problema: Problema) =
            context.startActivity(Intent(context, ErrorAnalisisActivity::class.java).putExtra(EXTRA_MOTIVO, problema.name))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityErrorBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()

        val problema = runCatching { Problema.valueOf(intent.getStringExtra(EXTRA_MOTIVO)!!) }.getOrDefault(Problema.SIN_PALTA)
        b.textoMotivo.setText(
            when (problema) {
                Problema.VARIAS_PALTAS -> R.string.motivo_varias
                Problema.PALTA_CORTADA -> R.string.motivo_cortada
                else -> R.string.motivo_sin_palta
            }
        )
        b.botonReintentar.setOnClickListener { finish(); MainActivity.volver(this, MainActivity.ACCION_CAMARA) }
        b.botonInicio.setOnClickListener { finish(); MainActivity.volver(this) }
    }
}
