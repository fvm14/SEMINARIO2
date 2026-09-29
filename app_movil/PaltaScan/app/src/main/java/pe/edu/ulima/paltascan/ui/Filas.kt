package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.card.MaterialCardView
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.databinding.ItemDiaBinding
import pe.edu.ulima.paltascan.databinding.ItemInspeccionBinding
import pe.edu.ulima.paltascan.databinding.VistaResumenBinding
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.datos.Resumen

/** Rellena la tarjeta de un analisis (Inicio, Historial y Lote). */
fun ItemInspeccionBinding.mostrar(i: Inspeccion, conFecha: Boolean = false) {
    val ctx = root.context
    etiquetaCategoria.text = Textos.CATEGORIAS_CORTAS[i.categoria]
    etiquetaCategoria.setTextColor(ContextCompat.getColor(ctx, Textos.colorCategoria(i.categoria)))
    etiquetaCategoria.backgroundTintList = ContextCompat.getColorStateList(ctx, Textos.fondoCategoria(i.categoria))
    textoHora.text = if (conFecha) Textos.fechaHora(i.fechaMs) else Textos.hora(i.fechaMs)
    textoMadurez.text = "${i.madurez} · ${Textos.nombreMadurez(i.madurez)}"
    textoDefecto.text = ctx.getString(R.string.defecto_pct, Textos.porcentaje(i.ratio))
    miniatura.setImageBitmap(BitmapFactory.decodeFile(i.archivoOverlay, BitmapFactory.Options().apply { inSampleSize = 8 }))
    root.setOnClickListener { ctx.startActivity(ResultadoActivity.intentDetalle(ctx, i.id)) }
}

/** Lista de analisis con encabezados por dia (Historial) o con una cabecera propia (Lote). */
class AdaptadorInspecciones(
    private val cabecera: ((ViewGroup) -> RecyclerView.ViewHolder)? = null,
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private sealed class Elemento {
        object Cabecera : Elemento()
        class Dia(val texto: String) : Elemento()
        class Fila(val i: Inspeccion) : Elemento()
    }

    private var elementos: List<Elemento> = emptyList()

    fun mostrar(context: Context, inspecciones: List<Inspeccion>, agruparPorDia: Boolean) {
        elementos = buildList {
            if (cabecera != null) add(Elemento.Cabecera)
            var diaActual = -1L
            for (i in inspecciones) {
                val dia = Textos.inicioDelDia(i.fechaMs)
                if (agruparPorDia && dia != diaActual) {
                    add(Elemento.Dia(Textos.encabezadoDia(context, i.fechaMs)))
                    diaActual = dia
                }
                add(Elemento.Fila(i))
            }
        }
        notifyDataSetChanged()
    }

    override fun getItemCount() = elementos.size

    override fun getItemViewType(pos: Int) = when (elementos[pos]) {
        Elemento.Cabecera -> 0
        is Elemento.Dia -> 1
        is Elemento.Fila -> 2
    }

    override fun onCreateViewHolder(parent: ViewGroup, tipo: Int): RecyclerView.ViewHolder {
        val inf = LayoutInflater.from(parent.context)
        return when (tipo) {
            0 -> cabecera!!(parent)
            1 -> object : RecyclerView.ViewHolder(ItemDiaBinding.inflate(inf, parent, false).root) {}
            else -> FilaVH(ItemInspeccionBinding.inflate(inf, parent, false))
        }
    }

    override fun onBindViewHolder(h: RecyclerView.ViewHolder, pos: Int) {
        when (val e = elementos[pos]) {
            is Elemento.Dia -> (h.itemView as TextView).text = e.texto
            is Elemento.Fila -> (h as FilaVH).v.mostrar(e.i)
            Elemento.Cabecera -> Unit
        }
    }

    class FilaVH(val v: ItemInspeccionBinding) : RecyclerView.ViewHolder(v.root)
}

/** Tarjetas de Cat. I / Cat. II / Rechazado (% y conteo) e histograma de madurez. */
fun VistaResumenBinding.mostrar(r: Resumen) {
    val ctx = root.context
    tarjetasCategoria.removeAllViews()
    for (c in 0 until 3) {
        val tarjeta = MaterialCardView(ctx).apply {
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                if (c > 0) marginStart = (8 * resources.displayMetrics.density).toInt()
            }
            setCardBackgroundColor(ContextCompat.getColor(ctx, Textos.fondoCategoria(c)))
            strokeWidth = 0
        }
        val columna = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            val p = (12 * resources.displayMetrics.density).toInt()
            setPadding(p, p, p, p)
        }
        val color = ContextCompat.getColor(ctx, Textos.colorCategoria(c))
        columna.addView(TextView(ctx).apply { text = Textos.CATEGORIAS_CORTAS[c]; setTextColor(color); textSize = 13f })
        columna.addView(TextView(ctx).apply {
            text = "${r.porcentaje(c)} %"; setTextColor(color); textSize = 24f; paint.isFakeBoldText = true
        })
        columna.addView(TextView(ctx).apply {
            text = ctx.getString(R.string.n_paltas, r.porCategoria[c]); setTextColor(color); textSize = 12f
        })
        tarjeta.addView(columna)
        tarjetasCategoria.addView(tarjeta)
    }
    histograma.mostrar(r.porMadurez)
}

fun Context.abrir(clase: Class<*>, configurar: Intent.() -> Unit = {}) =
    startActivity(Intent(this, clase).apply(configurar))
