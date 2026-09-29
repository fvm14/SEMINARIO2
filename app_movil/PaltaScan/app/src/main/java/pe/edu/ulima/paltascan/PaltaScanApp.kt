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

    /** La guia rapida se muestra solo la primera vez. */
    var guiaVista: Boolean
        get() = prefs.getBoolean("guia_vista", false)
        set(valor) = prefs.edit().putBoolean("guia_vista", valor).apply()

    /** Si es false, solo se conserva la imagen con marcas. */
    var guardarFotos: Boolean
        get() = prefs.getBoolean("guardar_fotos", true)
        set(valor) = prefs.edit().putBoolean("guardar_fotos", valor).apply()
}

val Context.app: PaltaScanApp get() = applicationContext as PaltaScanApp
