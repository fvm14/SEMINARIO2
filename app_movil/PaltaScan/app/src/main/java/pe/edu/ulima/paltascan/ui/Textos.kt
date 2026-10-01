package pe.edu.ulima.paltascan.ui

import pe.edu.ulima.paltascan.R
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Nombres, colores y formatos que comparten las pantallas. */
object Textos {
    val LOCALE: Locale = Locale.forLanguageTag("es-PE")

    val NOMBRES_MADUREZ = listOf("Poco maduro", "Poco-medio", "Medio", "Medio-maduro", "Muy maduro")
    val CATEGORIAS = listOf("Categoría I", "Categoría II", "Rechazado")

    fun nombreMadurez(nivel: Int) = NOMBRES_MADUREZ[nivel - 1]

    fun colorCategoria(categoria: Int) = when (categoria) {
        0 -> R.color.cat_i
        1 -> R.color.cat_ii
        else -> R.color.rechazo
    }

    /** 0.032 -> "3.2 %" */
    fun porcentaje(ratio: Double): String = "%.1f %%".format(Locale.US, ratio * 100)

    fun fechaHora(ms: Long): String = SimpleDateFormat("d MMM yyyy · HH:mm", LOCALE).format(Date(ms)).replace(".", "")
}
