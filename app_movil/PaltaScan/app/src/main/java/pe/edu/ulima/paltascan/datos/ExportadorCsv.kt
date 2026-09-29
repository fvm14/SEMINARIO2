package pe.edu.ulima.paltascan.datos

import pe.edu.ulima.paltascan.ml.Ocde
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ExportadorCsv {
    private val formato = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

    fun generar(inspecciones: List<Inspeccion>): String = buildString {
        appendLine("id,lote,fecha,madurez,prob_madurez,ratio_defecto,categoria_ocde,ms_preproceso,ms_inferencia,ms_postproceso,ms_total")
        for (i in inspecciones.sortedBy { it.fechaMs }) {
            appendLine(
                listOf(
                    i.id, csv(i.lote), formato.format(Date(i.fechaMs)), i.madurez,
                    "%.4f".format(Locale.US, i.probMadurez), "%.4f".format(Locale.US, i.ratio),
                    csv(Ocde.CATEGORIAS[i.categoria]),
                    "%.1f".format(Locale.US, i.msPreproceso), "%.1f".format(Locale.US, i.msInferencia),
                    "%.1f".format(Locale.US, i.msPostproceso), "%.1f".format(Locale.US, i.msTotal),
                ).joinToString(",")
            )
        }
    }

    private fun csv(s: String) = if (s.any { it == ',' || it == '"' || it == '\n' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
}
