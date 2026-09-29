package pe.edu.ulima.paltascan.ui

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import pe.edu.ulima.paltascan.R
import pe.edu.ulima.paltascan.app
import pe.edu.ulima.paltascan.databinding.ActivityResultadoBinding
import pe.edu.ulima.paltascan.datos.Inspeccion
import pe.edu.ulima.paltascan.ml.Ocde
import pe.edu.ulima.paltascan.ml.ValidacionCaptura.Problema
import java.io.File

/**
 * Resultado de un analisis recien hecho (p. 11-14) o detalle de uno del
 * historial (p. 23-24). Cambian el titulo y los botones:
 *  - nuevo:   Guardar en lote | Compartir (o Repetir foto si es poco confiable) | Nuevo analisis
 *  - detalle: Mover a lote | Eliminar | Compartir
 */
class ResultadoActivity : AppCompatActivity() {

    companion object {
        private const val EXTRA_ID = "id_inspeccion"
        private const val EXTRA_DETALLE = "detalle"

        fun abrirNuevo(context: Context, id: Long) =
            context.startActivity(Intent(context, ResultadoActivity::class.java).putExtra(EXTRA_ID, id))

        fun intentDetalle(context: Context, id: Long) =
            Intent(context, ResultadoActivity::class.java).putExtra(EXTRA_ID, id).putExtra(EXTRA_DETALLE, true)
    }

    private lateinit var b: ActivityResultadoBinding
    private lateinit var i: Inspeccion
    private val detalle get() = intent.getBooleanExtra(EXTRA_DETALLE, false)

    /** Al volver de "Nuevo lote" con un lote creado, se asigna el analisis. */
    private val crearLote = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { res ->
        val loteId = res.data?.getLongExtra(NuevoLoteActivity.EXTRA_LOTE_ID, -1L) ?: -1L
        if (loteId > 0) asignarLote(loteId)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        b = ActivityResultadoBinding.inflate(layoutInflater)
        setContentView(b.root)
        b.root.respetarBarrasDelSistema()
        b.barra.setNavigationOnClickListener { finish() }

        i = app.baseDatos.obtener(intent.getLongExtra(EXTRA_ID, -1)) ?: return finish()
        mostrar()
    }

    private fun mostrar() {
        val color = ContextCompat.getColor(this, Textos.colorCategoria(i.categoria))
        b.barra.setTitle(if (detalle) R.string.analisis else R.string.resultado)
        b.barra.subtitle = if (detalle) {
            Textos.fechaHora(i.fechaMs) + " · " + nombreLote()
        } else null

        // Imagen: original / con marcas
        val original = i.archivoFoto.takeIf { it.isNotEmpty() && File(it).exists() }
        b.botonOriginal.isEnabled = original != null
        b.selectorVista.addOnButtonCheckedListener { _, id, activo ->
            if (activo) mostrarImagen(if (id == R.id.botonOriginal) original else i.archivoOverlay)
        }
        mostrarImagen(i.archivoOverlay)

        // Categoria
        b.tarjetaCategoria.setCardBackgroundColor(ContextCompat.getColor(this, Textos.fondoCategoria(i.categoria)))
        b.textoCategoria.text = Ocde.CATEGORIAS[i.categoria]
        b.textoCategoria.setTextColor(color)
        b.textoSinDefectos.visibility = if (i.ratio == 0.0) View.VISIBLE else View.GONE
        b.textoRatio.text = Textos.porcentaje(i.ratio)
        b.barraDefecto.ratio = i.ratio

        // Madurez
        b.textoMadurez.text = getString(R.string.madurez_nivel, i.madurez, Textos.nombreMadurez(i.madurez))
        b.escalaMadurez.nivel = i.madurez
        val confianza = Math.round(i.probMadurez * 100)
        b.textoConfianza.text = getString(R.string.confianza, confianza)
        b.barraConfianza.setProgressCompat(confianza, false)
        b.textoTiempo.text = getString(R.string.tiempo_analisis, i.msTotal / 1000)

        // Avisos (p. 13)
        val avisos = i.avisos.split(",").filter { it.isNotBlank() }.mapNotNull { runCatching { Problema.valueOf(it) }.getOrNull() }
        val poco = avisos.isNotEmpty()
        b.panelAviso.visibility = if (poco) View.VISIBLE else View.GONE
        if (poco) {
            b.textoAviso.text = (listOf(getString(R.string.aviso_poco_confiable)) + avisos.map {
                getString(
                    when (it) {
                        Problema.OSCURA -> R.string.aviso_oscura
                        Problema.BORROSA -> R.string.aviso_borrosa
                        else -> R.string.aviso_confianza
                    }
                )
            }).joinToString("\n")
        }

        // Botones
        if (detalle) {
            b.botonSecundario1.setText(R.string.mover_a_lote)
            b.botonSecundario1.setIconResource(R.drawable.ic_carpeta)
            b.botonSecundario1.setOnClickListener { elegirLote() }
            b.botonSecundario2.setText(R.string.eliminar)
            b.botonSecundario2.setIconResource(R.drawable.ic_eliminar)
            b.botonSecundario2.setTextColor(ContextCompat.getColor(this, R.color.rechazo))
            b.botonSecundario2.iconTint = ContextCompat.getColorStateList(this, R.color.rechazo)
            b.botonSecundario2.setOnClickListener { Acciones.confirmarEliminar(this, i) { finish() } }
            b.botonPrincipal.setText(R.string.compartir)
            b.botonPrincipal.setIconResource(R.drawable.ic_compartir)
            b.botonPrincipal.setOnClickListener { compartir() }
        } else {
            b.botonSecundario1.setText(R.string.guardar_en_lote)
            b.botonSecundario1.setOnClickListener { elegirLote() }
            if (poco) {
                b.botonSecundario2.setText(R.string.repetir_foto)
                b.botonSecundario2.setIconResource(R.drawable.ic_camara)
                b.botonSecundario2.setOnClickListener { finish(); MainActivity.volver(this, MainActivity.ACCION_CAMARA) }
            } else {
                b.botonSecundario2.setText(R.string.compartir)
                b.botonSecundario2.setOnClickListener { compartir() }
            }
            b.botonPrincipal.setOnClickListener { finish(); MainActivity.volver(this, MainActivity.ACCION_CAMARA) }
        }
    }

    private fun mostrarImagen(ruta: String?) {
        ruta ?: return
        b.imagen.setImageBitmap(BitmapFactory.decodeFile(ruta))
        b.leyenda.visibility = if (ruta == i.archivoOverlay) View.VISIBLE else View.GONE
    }

    private fun nombreLote(): String =
        i.loteId?.let { app.baseDatos.lote(it)?.nombre } ?: getString(R.string.sin_lote)

    private fun elegirLote() {
        Acciones.elegirLote(this, permitirQuitar = detalle && i.loteId != null,
            alElegir = { asignarLote(it) },
            alPedirNuevo = { crearLote.launch(Intent(this, NuevoLoteActivity::class.java).putExtra(NuevoLoteActivity.EXTRA_DEVOLVER, true)) })
    }

    private fun asignarLote(loteId: Long?) {
        app.baseDatos.moverALote(i.id, loteId)
        i = app.baseDatos.obtener(i.id) ?: return
        Toast.makeText(this, getString(R.string.guardado_en, nombreLote()), Toast.LENGTH_SHORT).show()
        if (detalle) b.barra.subtitle = Textos.fechaHora(i.fechaMs) + " · " + nombreLote()
    }

    private fun compartir() {
        val texto = getString(
            R.string.compartir_texto, Ocde.CATEGORIAS[i.categoria], i.madurez,
            Textos.nombreMadurez(i.madurez), Textos.porcentaje(i.ratio)
        )
        Acciones.compartirArchivo(this, File(i.archivoOverlay), "image/jpeg", texto)
    }
}
