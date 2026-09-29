package pe.edu.ulima.paltascan.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.FragmentLotesBinding
import pe.edu.ulima.paltascan.databinding.ItemLoteBinding
import pe.edu.ulima.paltascan.datos.Lote
import pe.edu.ulima.paltascan.datos.Resumen

/** Lotes (p. 25-26): lista con la proporcion por categoria de cada lote. */
class LotesFragment : Fragment() {

    private var _b: FragmentLotesBinding? = null
    private val b get() = _b!!

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        _b = FragmentLotesBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, s: Bundle?) {
        val ctx = requireContext()
        b.lista.layoutManager = LinearLayoutManager(ctx)
        b.botonNuevoLote.setOnClickListener { ctx.abrir(NuevoLoteActivity::class.java) }
        b.vacio.iconoVacio.setImageResource(R.drawable.ic_lotes)
        b.vacio.tituloVacio.setText(R.string.lotes_vacio_titulo)
        b.vacio.textoVacio.setText(R.string.lotes_vacio_texto)
        b.vacio.botonVacio.setText(R.string.nuevo_lote)
        b.vacio.botonVacio.setOnClickListener { ctx.abrir(NuevoLoteActivity::class.java) }
        for (c in 0 until 3) {
            b.leyenda.addView(TextView(ctx).apply {
                text = "●  " + Textos.CATEGORIAS_CORTAS[c]
                setTextColor(ContextCompat.getColor(ctx, Textos.colorCategoria(c)))
                textSize = 13f
                setPadding(0, 0, (18 * resources.displayMetrics.density).toInt(), 0)
            })
        }
    }

    override fun onResume() {
        super.onResume()
        val bd = requireContext().app.baseDatos
        val lotes = bd.lotes().map { it to Resumen(bd.porLote(it.id)) }
        b.lista.adapter = Adaptador(lotes)
        val vacio = lotes.isEmpty()
        b.vacio.root.visibility = if (vacio) View.VISIBLE else View.GONE
        b.botonNuevoLote.visibility = if (vacio) View.GONE else View.VISIBLE
        b.leyenda.visibility = if (vacio) View.GONE else View.VISIBLE
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _b = null
    }

    private class Adaptador(val lotes: List<Pair<Lote, Resumen>>) : RecyclerView.Adapter<Adaptador.VH>() {
        class VH(val v: ItemLoteBinding) : RecyclerView.ViewHolder(v.root)

        override fun onCreateViewHolder(parent: ViewGroup, t: Int) =
            VH(ItemLoteBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = lotes.size

        override fun onBindViewHolder(h: VH, pos: Int) {
            val (lote, r) = lotes[pos]
            val ctx = h.itemView.context
            h.v.textoNombre.text = lote.nombre
            h.v.textoDetalle.text = Textos.fecha(lote.fechaMs) + " · " + ctx.resources.getQuantityString(R.plurals.n_paltas, r.total, r.total)
            h.v.barra.mostrar(r.porCategoria)
            h.v.textoPorcentajes.text = (0 until 3).joinToString(" · ") { "${r.porcentaje(it)} %" }
            h.v.root.setOnClickListener { LoteActivity.abrir(ctx, lote.id) }
        }
    }
}
