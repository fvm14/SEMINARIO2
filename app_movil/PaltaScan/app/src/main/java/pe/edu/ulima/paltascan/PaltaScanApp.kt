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

    /**
     * Ejecutar el modelo en la GPU en vez de la CPU (por defecto, CPU). En la version
     * ligera no se ofrece la opcion: preparar el delegado GPU tarda ~14 s en cada
     * arranque (compilacion OpenCL en el Xiaomi 11 Lite 5G NE), y la CPU arranca al instante.
     */
    var usarGpu: Boolean
        get() = gpuDisponible && BuildConfig.OPCIONES_TECNICAS && prefs.getBoolean("usar_gpu", false)
        set(valor) = prefs.edit().putBoolean("usar_gpu", valor).apply()

    /** Evaluar el fruto completo con dos fotos, lado A y lado B (por defecto, si). */
    var dosCaras: Boolean
        get() = prefs.getBoolean("dos_caras", true)
        set(valor) = prefs.edit().putBoolean("dos_caras", valor).apply()

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
