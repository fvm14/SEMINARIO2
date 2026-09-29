package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivitySplashBinding

/** Pantalla de arranque (p. 1): carga el modelo antes de mostrar la app. */
class SplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivitySplashBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()

        lifecycleScope.launch {
            val listo = withContext(Dispatchers.Default) { app.analizador != null }
            if (!listo) {
                b.progreso.visibility = View.GONE
                b.textoEstado.setText(R.string.modelo_faltante)
                return@launch
            }
            val destino = if (app.guiaVista) MainActivity::class.java else GuiaActivity::class.java
            startActivity(Intent(this@SplashActivity, destino))
            finish()
        }
    }
}
