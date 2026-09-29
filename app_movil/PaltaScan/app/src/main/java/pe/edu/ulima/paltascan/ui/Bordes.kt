package pe.edu.ulima.paltascan.ui

import android.view.View
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding

/** Android 15+ dibuja detras de las barras del sistema: se suma su tamano al padding original. */
fun View.respetarBarrasDelSistema() {
    val izq = paddingLeft
    val arriba = paddingTop
    val der = paddingRight
    val abajo = paddingBottom
    ViewCompat.setOnApplyWindowInsetsListener(this) { v, insets ->
        val b = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
        v.updatePadding(izq + b.left, arriba + b.top, der + b.right, abajo + b.bottom)
        insets
    }
}
