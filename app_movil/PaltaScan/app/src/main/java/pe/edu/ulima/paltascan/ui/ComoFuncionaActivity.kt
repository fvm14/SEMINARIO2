package pe.edu.ulima.paltascan.ui

import android.os.Bundle
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.google.android.material.card.MaterialCardView
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.databinding.ActivityComoFuncionaBinding

/** Como funciona (p. 31): categorias OCDE y escala de madurez. */
class ComoFuncionaActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val b = ActivityComoFuncionaBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.barra.setNavigationOnClickListener { finish() }

        val dp = resources.displayMetrics.density
        val limites = listOf(R.string.cf_cat_i, R.string.cf_cat_ii, R.string.cf_rechazo)
        for (c in 0 until 3) {
            val color = ContextCompat.getColor(this, Textos.colorCategoria(c))
            val tarjeta = MaterialCardView(this).apply {
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                    if (c > 0) marginStart = (8 * dp).toInt()
                }
                setCardBackgroundColor(ContextCompat.getColor(context, Textos.fondoCategoria(c)))
                strokeWidth = 0
            }
            val col = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                setPadding((12 * dp).toInt(), (12 * dp).toInt(), (12 * dp).toInt(), (12 * dp).toInt())
            }
            col.addView(TextView(this).apply { text = Textos.CATEGORIAS_CORTAS[c]; setTextColor(color); textSize = 15f; paint.isFakeBoldText = true })
            col.addView(TextView(this).apply { setText(limites[c]); setTextColor(color); textSize = 12f })
            tarjeta.addView(col)
            b.categorias.addView(tarjeta)
        }

        for (n in 1..5) {
            val fila = layoutInflater.inflate(android.R.layout.simple_list_item_1, b.niveles, false) as TextView
            fila.text = "$n  ·  ${Textos.nombreMadurez(n)}"
            fila.setTextColor(ContextCompat.getColor(this, R.color.texto))
            b.niveles.addView(fila)
        }
    }
}
