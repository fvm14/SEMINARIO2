package pe.edu.ulima.paltascan

import android.app.Application
import android.content.Context
import pe.edu.ulima.paltascan.datos.BaseDatos
import pe.edu.ulima.paltascan.ml.Analizador

class PaltaScanApp : Application() {

    val baseDatos by lazy { BaseDatos(this) }

    /** null si el modelo no esta en assets (ver app_movil/README.md). */
    val analizador: Analizador? by lazy {
        if (Analizador.existeModelo(this)) Analizador.cargar(this) else null
    }

    private val prefs by lazy { getSharedPreferences("paltascan", Context.MODE_PRIVATE) }

    /** Lote donde se guardan los nuevos analisis; null = sin lote. */
    var loteActualId: Long?
        get() = prefs.getLong("lote_actual_id", -1L).takeIf { it > 0 && baseDatos.lote(it) != null }
        set(valor) = prefs.edit().putLong("lote_actual_id", valor ?: -1L).apply()
}

val Context.app: PaltaScanApp get() = applicationContext as PaltaScanApp
