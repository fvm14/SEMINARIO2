package pe.edu.ulima.paltascan.ui

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import pe.edu.ulima.paltascan.databinding.ActivityInfoBinding

/** Como funciona: que hace la app, la norma OCDE, el calculo, la madurez y consejos para la foto. */
class InfoActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityInfoBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.botonAtras.setOnClickListener { finish() }
    }
}
