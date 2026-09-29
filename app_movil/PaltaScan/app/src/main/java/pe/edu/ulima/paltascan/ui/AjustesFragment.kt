package pe.edu.ulima.paltascan.ui

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import pe.edu.ulima.paltascan.BuildConfig
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.FragmentAjustesBinding

/** Ajustes (p. 30). */
class AjustesFragment : Fragment() {

    private var _b: FragmentAjustesBinding? = null
    private val b get() = _b!!

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, s: Bundle?): View {
        _b = FragmentAjustesBinding.inflate(inflater, container, false)
        return b.root
    }

    override fun onViewCreated(view: View, s: Bundle?) {
        val ctx = requireContext()
        b.interruptorFotos.isChecked = ctx.app.guardarFotos
        b.interruptorFotos.setOnCheckedChangeListener { _, activo -> ctx.app.guardarFotos = activo }
        b.textoVersion.text = getString(R.string.version_modelo_valor) + " · app " + BuildConfig.VERSION_NAME
        b.filaComoFunciona.setOnClickListener { ctx.abrir(ComoFuncionaActivity::class.java) }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _b = null
    }
}
