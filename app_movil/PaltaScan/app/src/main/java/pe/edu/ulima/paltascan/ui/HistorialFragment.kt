package pe.edu.ulima.paltascan.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.FragmentHistorialBinding
import pe.edu.ulima.paltascan.datos.Filtro

/** Historial (p. 21-22): analisis agrupados por dia, con filtros. */
class HistorialFragment : Fragment() {

    private var _b: FragmentHistorialBinding? = null
    private val b get() = _b!!
    private val adaptador = AdaptadorInspecciones()
    private var filtro = Filtro()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        _b = FragmentHistorialBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, s: Bundle?) {
        b.lista.layoutManager = LinearLayoutManager(requireContext())
        b.lista.adapter = adaptador
        b.vacio.iconoVacio.setImageResource(R.drawable.ic_historial)
        b.vacio.tituloVacio.setText(R.string.historial_vacio_titulo)
        b.vacio.textoVacio.setText(R.string.historial_vacio_texto)
        b.vacio.botonVacio.setText(R.string.analizar_palta)
        b.vacio.botonVacio.setOnClickListener { (requireActivity() as MainActivity).abrirCamara() }

        b.chipTodas.setOnClickListener { filtro = Filtro(); b.chipMadurez.setText(R.string.filtro_madurez); cargar() }
        b.chipCatI.setOnClickListener { filtro = Filtro(categoria = 0); cargar() }
        b.chipCatII.setOnClickListener { filtro = Filtro(categoria = 1); cargar() }
        b.chipRechazado.setOnClickListener { filtro = Filtro(categoria = 2); cargar() }
        b.chipMadurez.setOnClickListener { elegirMadurez() }
    }

    override fun onResume() {
        super.onResume()
        cargar()
    }

    private fun elegirMadurez() {
        val opciones = Textos.NOMBRES_MADUREZ.mapIndexed { i, n -> "${i + 1} · $n" }.toTypedArray()
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.filtro_madurez)
            .setItems(opciones) { _, i ->
                filtro = Filtro(madurez = i + 1)
                b.chipMadurez.text = getString(R.string.filtro_madurez) + " " + (i + 1)
                cargar()
            }
            .setOnCancelListener { if (filtro.madurez == null) b.chipTodas.isChecked = true }
            .show()
    }

    private fun cargar() {
        val bd = requireContext().app.baseDatos
        val lista = bd.historial(filtro)
        adaptador.mostrar(requireContext(), lista, agruparPorDia = true)
        // El estado vacio solo aparece si no hay ningun analisis; con filtro se ve la lista vacia.
        val sinNada = lista.isEmpty() && bd.contarInspecciones() == 0
        b.vacio.root.visibility = if (sinNada) View.VISIBLE else View.GONE
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _b = null
    }
}
