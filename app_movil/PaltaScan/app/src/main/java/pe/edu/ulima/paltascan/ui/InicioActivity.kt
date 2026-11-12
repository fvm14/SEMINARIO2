package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.os.Bundle
import android.os.SystemClock
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityInicioBinding

/**
 * Pantalla de inicio con el logo y el nombre. Carga el modelo mientras se
 * muestra (la pantalla principal lo reutiliza) y luego pasa a MainActivity.
 */
class InicioActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityInicioBinding.inflate(layoutInflater)
        setContentView(b.root)

        b.logo.alpha = 0f
        b.nombre.alpha = 0f
        b.lema.alpha = 0f
        b.progreso.alpha = 0f
        b.textoCarga.alpha = 0f
        b.logo.animate().alpha(1f).setDuration(350).start()
        b.nombre.animate().alpha(1f).setStartDelay(150).setDuration(350).start()
        b.lema.animate().alpha(1f).setStartDelay(300).setDuration(350).start()
        b.progreso.animate().alpha(1f).setStartDelay(450).setDuration(300).start()
        b.textoCarga.animate().alpha(1f).setStartDelay(450).setDuration(300).start()

        val inicio = SystemClock.elapsedRealtime()
        lifecycleScope.launch {
            withContext(Dispatchers.Default) { app.analizador }
            val restante = MINIMO_MS - (SystemClock.elapsedRealtime() - inicio)
            if (restante > 0) delay(restante)
            startActivity(Intent(this@InicioActivity, MainActivity::class.java))
            @Suppress("DEPRECATION")
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
        }
    }

    companion object {
        /** Tiempo minimo en pantalla, para que el logo no aparezca solo un instante. */
        private const val MINIMO_MS = 1600L
    }
}
