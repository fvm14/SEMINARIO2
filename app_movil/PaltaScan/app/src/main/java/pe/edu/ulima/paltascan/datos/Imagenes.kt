package pe.edu.ulima.paltascan.datos

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import pe.edu.ulima.paltascan.ml.Ocde
import pe.edu.ulima.paltascan.ml.ResultadoAnalisis
import java.io.File
import java.io.FileOutputStream

object Imagenes {
    private const val LADO_MAXIMO = 1600

    private val COLOR_FRUTO = Color.argb(80, 46, 160, 67)
    private val COLOR_DEFECTO = Color.argb(170, 220, 38, 38)

    // Camara en vivo: rellenos suaves para que se vea la palta debajo
    private val COLOR_FRUTO_VIVO = Color.argb(45, 102, 187, 106)
    private val COLOR_BORDE_FRUTO = Color.rgb(102, 187, 106)
    private val COLOR_DEFECTO_VIVO = Color.argb(110, 229, 57, 53)
    private val COLOR_BORDE_DEFECTO = Color.rgb(255, 82, 82)

    /** Carga la foto reducida (lado <= 1600 px) y con la rotacion EXIF aplicada. */
    fun cargar(context: Context, uri: Uri): Bitmap {
        val cr = context.contentResolver
        val limites = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        cr.openInputStream(uri).use { BitmapFactory.decodeStream(it, null, limites) }
        var muestreo = 1
        while (maxOf(limites.outWidth, limites.outHeight) / (muestreo * 2) >= LADO_MAXIMO) muestreo *= 2
        val bmp = cr.openInputStream(uri).use {
            BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = muestreo })
        } ?: error("No se pudo leer la imagen")
        val grados = cr.openInputStream(uri).use { ExifInterface(it!!).rotationDegrees }
        val girada = if (grados == 0) bmp else Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, Matrix().apply { postRotate(grados.toFloat()) }, true)
        val escala = LADO_MAXIMO.toFloat() / maxOf(girada.width, girada.height)
        return if (escala >= 1f) girada else Bitmap.createScaledBitmap(girada, (girada.width * escala).toInt(), (girada.height * escala).toInt(), true)
    }

    /** Foto con el fruto (verde) y los defectos (rojo) superpuestos. */
    fun superponer(foto: Bitmap, r: ResultadoAnalisis): Bitmap {
        val m = r.mascaras
        val px = IntArray(m.ancho * m.alto) { i ->
            when {
                r.defecto[i] -> COLOR_DEFECTO
                r.fruto[i] -> COLOR_FRUTO
                else -> Color.TRANSPARENT
            }
        }
        val capa = Bitmap.createBitmap(px, m.ancho, m.alto, Bitmap.Config.ARGB_8888)
        val salida = foto.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(salida).drawBitmap(capa, null, RectF(0f, 0f, salida.width.toFloat(), salida.height.toFloat()), Paint())
        capa.recycle()
        return salida
    }

    /**
     * Capa transparente para la camara en vivo, del tamano de las mascaras:
     * todas las paltas detectadas con contorno verde y los defectos en rojo
     * (relleno y contorno). Con una sola palta se dibujan el fruto y los
     * defectos que entran al ratio; si no, lo que entrego el modelo.
     */
    fun contornos(r: ResultadoAnalisis): Bitmap {
        val m = r.mascaras
        val n = m.ancho * m.alto
        val palta = BooleanArray(n) { m.palta[it] || r.fruto[it] }
        val defecto = if (r.bloqueante == null) r.defecto else m.defecto
        val bordePalta = borde(palta, m.ancho, m.alto)
        val bordeDefecto = borde(defecto, m.ancho, m.alto)
        val px = IntArray(n) { i ->
            when {
                bordeDefecto[i] -> COLOR_BORDE_DEFECTO
                defecto[i] -> COLOR_DEFECTO_VIVO
                bordePalta[i] -> COLOR_BORDE_FRUTO
                palta[i] -> COLOR_FRUTO_VIVO
                else -> Color.TRANSPARENT
            }
        }
        return Bitmap.createBitmap(px, m.ancho, m.alto, Bitmap.Config.ARGB_8888)
    }

    /** Pixeles de la mascara a 2 px o menos de su exterior. */
    private fun borde(mascara: BooleanArray, ancho: Int, alto: Int): BooleanArray {
        val exterior = Ocde.dilatar(BooleanArray(mascara.size) { !mascara[it] }, ancho, alto, 5)
        return BooleanArray(mascara.size) { mascara[it] && exterior[it] }
    }

    fun guardar(context: Context, bmp: Bitmap, nombre: String): String {
        val dir = File(context.filesDir, "inspecciones").apply { mkdirs() }
        val archivo = File(dir, nombre)
        FileOutputStream(archivo).use { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }
        return archivo.absolutePath
    }
}
