package pe.edu.ulima.paltascan.ui

import android.content.Context
import pe.edu.ulima.paltascan.R
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/** Nombres, colores y formatos que comparten todas las pantallas. */
object Textos {
    val LOCALE: Locale = Locale.forLanguageTag("es-PE")

    val NOMBRES_MADUREZ = listOf("Poco maduro", "Poco-medio", "Medio", "Medio-maduro", "Muy maduro")
    val CATEGORIAS_CORTAS = listOf("Cat. I", "Cat. II", "Rechazado")

    fun nombreMadurez(nivel: Int) = NOMBRES_MADUREZ[nivel - 1]

    fun colorCategoria(categoria: Int) = when (categoria) {
        0 -> R.color.cat_i
        1 -> R.color.cat_ii
        else -> R.color.rechazo
    }

    fun fondoCategoria(categoria: Int) = when (categoria) {
        0 -> R.color.cat_i_fondo
        1 -> R.color.cat_ii_fondo
        else -> R.color.rechazo_fondo
    }

    /** 0.032 -> "3,2 %" */
    fun porcentaje(ratio: Double): String = "%.1f %%".format(LOCALE, ratio * 100)

    fun hora(ms: Long): String = SimpleDateFormat("HH:mm", LOCALE).format(Date(ms))

    fun fecha(ms: Long): String = SimpleDateFormat("d MMM yyyy", LOCALE).format(Date(ms)).replace(".", "")

    fun fechaHora(ms: Long): String = fecha(ms) + " · " + hora(ms)

    /** "Jueves 24 de septiembre" */
    fun fechaLarga(ms: Long): String =
        SimpleDateFormat("EEEE d 'de' MMMM", LOCALE).format(Date(ms)).replaceFirstChar { it.titlecase(LOCALE) }

    fun inicioDelDia(ms: Long): Long = Calendar.getInstance().apply {
        timeInMillis = ms
        set(Calendar.HOUR_OF_DAY, 0); set(Calendar.MINUTE, 0); set(Calendar.SECOND, 0); set(Calendar.MILLISECOND, 0)
    }.timeInMillis

    /** "Hoy · 24 sep 2026", "Ayer · 23 sep 2026" o solo la fecha. */
    fun encabezadoDia(context: Context, ms: Long): String {
        val hoy = inicioDelDia(System.currentTimeMillis())
        val dia = inicioDelDia(ms)
        val prefijo = when (dia) {
            hoy -> context.getString(R.string.hoy) + " · "
            hoy - 86_400_000L -> context.getString(R.string.ayer) + " · "
            else -> ""
        }
        return prefijo + fecha(ms)
    }

    fun saludo(context: Context): String {
        val h = Calendar.getInstance().get(Calendar.HOUR_OF_DAY)
        return context.getString(
            when {
                h < 12 -> R.string.buenos_dias
                h < 19 -> R.string.buenas_tardes
                else -> R.string.buenas_noches
            }
        )
    }
}
