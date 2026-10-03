package pe.edu.ulima.paltascan

import android.app.Application
import android.content.Context
import pe.edu.ulima.paltascan.datos.BaseDatos
import pe.edu.ulima.paltascan.ml.Analizador
import pe.edu.ulima.paltascan.ml.Configuracion
import pe.edu.ulima.paltascan.ml.ModeloApp

class PaltaScanApp : Application() {

    val baseDatos by lazy { BaseDatos(this) }

    private val prefs by lazy { getSharedPreferences("paltascan", Context.MODE_PRIVATE) }

    /** Modelos cuyos archivos estan en assets (ver app_movil/README.md). */
    val modelosDisponibles: List<ModeloApp> by lazy { Configuracion.MODELOS.filter { Analizador.existeModelo(this, it) } }

    /** Modelo con el que se analizan las fotos (por defecto, el propuesto). */
    var modeloActual: ModeloApp
        get() = Configuracion.modelo(prefs.getString("modelo_id", null)).takeIf { it in modelosDisponibles }
            ?: modelosDisponibles.firstOrNull() ?: Configuracion.MODELO_POR_DEFECTO
        set(valor) = prefs.edit().putString("modelo_id", valor.id).apply()

    /** Si el telefono admite el delegado GPU (si no, no se ofrece la opcion). */
    val gpuDisponible: Boolean by lazy { Analizador.gpuDisponible() }

    /** Ejecutar el modelo en la GPU en vez de la CPU (por defecto, CPU). */
    var usarGpu: Boolean
        get() = gpuDisponible && prefs.getBoolean("usar_gpu", false)
        set(valor) = prefs.edit().putBoolean("usar_gpu", valor).apply()

    private var cargado: Analizador? = null

    /** null si el modelo no esta en assets. Al cambiar de modelo o de procesador se libera el anterior. */
    val analizador: Analizador?
        @Synchronized get() {
            val modelo = modeloActual
            val gpu = usarGpu
            cargado?.let { if (it.modelo == modelo && it.pidioGpu == gpu) return it else it.cerrar() }
            cargado = if (Analizador.existeModelo(this, modelo)) Analizador.cargar(this, modelo, gpu) else null
            return cargado
        }

    /** Lote donde se guardan los nuevos analisis; null = sin lote. */
    var loteActualId: Long?
        get() = prefs.getLong("lote_actual_id", -1L).takeIf { it > 0 && baseDatos.lote(it) != null }
        set(valor) = prefs.edit().putLong("lote_actual_id", valor ?: -1L).apply()
}

val Context.app: PaltaScanApp get() = applicationContext as PaltaScanApp
