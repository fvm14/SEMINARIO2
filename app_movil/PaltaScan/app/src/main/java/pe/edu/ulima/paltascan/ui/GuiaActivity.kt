package pe.edu.ulima.paltascan.ui

import android.content.Intent
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityGuiaBinding

/** Guia rapida de 3 pasos (p. 3-5), solo la primera vez o desde Ajustes. */
class GuiaActivity : AppCompatActivity() {

    private data class Paso(val icono: Int, val titulo: Int, val texto: Int)

    private val pasos = listOf(
        Paso(R.drawable.ic_palta, R.string.guia_1_titulo, R.string.guia_1_texto),
        Paso(R.drawable.ic_sol, R.string.guia_2_titulo, R.string.guia_2_texto),
        Paso(R.drawable.ic_encuadre, R.string.guia_3_titulo, R.string.guia_3_texto),
    )
    private var paso = 0
    private lateinit var b: ActivityGuiaBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityGuiaBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        paso = savedInstanceState?.getInt("paso") ?: 0
        b.botonOmitir.setOnClickListener { terminar() }
        b.botonSiguiente.setOnClickListener { if (paso < pasos.lastIndex) { paso++; mostrar() } else terminar() }
        mostrar()
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putInt("paso", paso)
    }

    private fun mostrar() {
        val p = pasos[paso]
        b.textoPaso.text = getString(R.string.guia_paso, paso + 1)
        b.barraPasos.setProgressCompat(paso + 1, true)
        b.ilustracion.setImageResource(p.icono)
        // El icono de la palta ya tiene sus colores; los demas se tiñen de verde.
        b.ilustracion.imageTintList = if (p.icono == R.drawable.ic_palta) null else getColorStateList(R.color.verde_palta)
        b.textoTitulo.setText(p.titulo)
        b.textoDetalle.setText(p.texto)
        b.botonSiguiente.setText(if (paso == pasos.lastIndex) R.string.empezar else R.string.siguiente)
    }

    private fun terminar() {
        val primeraVez = !app.guiaVista
        app.guiaVista = true
        if (primeraVez) startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}
