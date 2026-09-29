package pe.edu.ulima.paltascan

import android.app.Application
import android.content.Context
import pe.edu.ulima.paltascan.datos.BaseDatos
import pe.edu.ulima.paltascan.ml.Analizador
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PaltaScanApp : Application() {

    val baseDatos by lazy { BaseDatos(this) }

    /** null si el modelo no esta en assets (ver app_movil/README.md). */
    val analizador: Analizador? by lazy {
        if (Analizador.existeModelo(this)) Analizador.cargar(this) else null
    }

    private val prefs by lazy { getSharedPreferences("paltascan", Context.MODE_PRIVATE) }

    var loteActual: String
        get() = prefs.getString("lote_actual", null) ?: nuevoLote()
        set(valor) = prefs.edit().putString("lote_actual", valor).apply()

    fun nuevoLote(): String {
        val nombre = "Lote " + SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).format(Date())
        loteActual = nombre
        return nombre
    }
}

val Context.app: PaltaScanApp get() = applicationContext as PaltaScanApp
