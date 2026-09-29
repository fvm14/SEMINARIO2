package pe.edu.ulima.paltascan.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.FragmentInicioBinding
import pe.edu.ulima.paltascan.databinding.ItemInspeccionBinding

/** Inicio (p. 2): saludo, ultimos analisis, accesos y botones de captura. */
class InicioFragment : Fragment() {

    private var _b: FragmentInicioBinding? = null
    private val b get() = _b!!
    private val main get() = requireActivity() as MainActivity

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        _b = FragmentInicioBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, s: Bundle?) {
        b.botonAnalizar.setOnClickListener { main.abrirCamara() }
        b.botonGaleria.setOnClickListener { main.abrirGaleria() }
        b.botonContinuo.setOnClickListener {
            Toast.makeText(requireContext(), R.string.modo_continuo_pronto, Toast.LENGTH_SHORT).show()
        }
        b.botonVerTodo.setOnClickListener { main.irA(R.id.nav_historial) }
        b.tarjetaHistorial.setOnClickListener { main.irA(R.id.nav_historial) }
        b.tarjetaLotes.setOnClickListener { main.irA(R.id.nav_lotes) }
    }

    override fun onResume() {
        super.onResume()
        val ctx = requireContext()
        b.textoSaludo.text = Textos.saludo(ctx)
        b.textoFecha.text = Textos.fechaLarga(System.currentTimeMillis())
        val bd = ctx.app.baseDatos
        val recientes = bd.historial(limite = 3)
        b.listaRecientes.removeAllViews()
        for (i in recientes) {
            ItemInspeccionBinding.inflate(layoutInflater, b.listaRecientes, true).mostrar(i)
        }
        b.textoSinAnalisis.visibility = if (recientes.isEmpty()) View.VISIBLE else View.GONE
        b.botonVerTodo.visibility = if (recientes.isEmpty()) View.GONE else View.VISIBLE
        b.textoNumAnalisis.text = getString(R.string.n_analisis, bd.contarInspecciones())
        b.textoNumLotes.text = bd.contarLotes().let { resources.getQuantityString(R.plurals.n_lotes, it, it) }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _b = null
    }
}
