package pe.edu.ulima.paltascan.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class Tiempos(val preprocesoMs: Double, val inferenciaMs: Double, val postprocesoMs: Double) {
    val totalMs: Double get() = preprocesoMs + inferenciaMs + postprocesoMs
}

class ResultadoAnalisis(
    val categoria: Int,
    val ratio: Double,
    val madurez: Int,
    val probMadurez: Float,
    val probabilidades: FloatArray,
    val mascaras: Mascaras,
    val fruto: BooleanArray,
    val defecto: BooleanArray,
    val tiempos: Tiempos,
) {
    val nombreCategoria: String get() = Ocde.CATEGORIAS[categoria]
    val frutoDetectado: Boolean get() = fruto.any { it }
}

/** Pipeline completo de inferencia_movil.analizar() sobre un Bitmap. */
class Analizador private constructor(private val interprete: Interpreter) {

    private val entrada = interprete.getInputTensor(0)
    private val formaEntrada = entrada.shape()
    private val nhwc = formaEntrada[3] == 3
    val lienzo: Int = if (nhwc) formaEntrada[1] else formaEntrada[2]

    companion object {
        fun cargar(context: Context, archivo: String = Configuracion.ARCHIVO_MODELO): Analizador {
            val fd = context.assets.openFd(archivo)
            val modelo = FileInputStream(fd.fileDescriptor).channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            val opciones = Interpreter.Options().setNumThreads(Configuracion.HILOS_CPU)
            return Analizador(Interpreter(modelo, opciones))
        }

        fun existeModelo(context: Context, archivo: String = Configuracion.ARCHIVO_MODELO): Boolean =
            context.assets.list("")?.contains(archivo) == true
    }

    fun analizar(foto: Bitmap): ResultadoAnalisis {
        val t0 = System.nanoTime()
        val lb = Letterbox.calcular(foto.height, foto.width, lienzo)
        val buffer = prepararEntrada(foto, lb)

        val t1 = System.nanoTime()
        val salidas = HashMap<Int, Any>()
        val buffersSalida = (0 until interprete.outputTensorCount).map { i ->
            val t = interprete.getOutputTensor(i)
            ByteBuffer.allocateDirect(t.numBytes()).order(ByteOrder.nativeOrder()).also { salidas[i] = it }
        }
        interprete.runForMultipleInputsOutputs(arrayOf(buffer), salidas)

        val t2 = System.nanoTime()
        val tensores = buffersSalida.mapIndexed { i, b -> leerTensor(i, b) }
        val salida = LectorSalidas.leer(tensores, lienzo)
        val instancias = Postproceso.seleccionar(salida)
        val m = Postproceso.mascaras(salida, instancias, lb)
        val fruto = Ocde.frutoCompleto(m.palta, m.defecto, m.ancho, m.alto)
        val defecto = Ocde.filtrarDefectoPorRoi(fruto, m.defecto, m.ancho, m.alto, lb.kernelRoi(Configuracion.KERNEL_ROI_PX))
        val ratio = Ocde.calcularRatio(fruto, defecto)
        var mejor = 0
        for (i in salida.madurez.indices) if (salida.madurez[i] > salida.madurez[mejor]) mejor = i
        val t3 = System.nanoTime()

        return ResultadoAnalisis(
            categoria = Ocde.clasificar(ratio),
            ratio = ratio,
            madurez = mejor + 1,
            probMadurez = salida.madurez[mejor],
            probabilidades = salida.madurez,
            mascaras = m,
            fruto = fruto,
            defecto = defecto,
            tiempos = Tiempos((t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6),
        )
    }

    private fun prepararEntrada(foto: Bitmap, lb: Letterbox): ByteBuffer {
        val lienzoBmp = Bitmap.createBitmap(lienzo, lienzo, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(lienzoBmp)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val destino = RectF(lb.left.toFloat(), lb.top.toFloat(), (lb.left + lb.nw).toFloat(), (lb.top + lb.nh).toFloat())
        canvas.drawBitmap(foto, null, destino, Paint(Paint.FILTER_BITMAP_FLAG))
        val px = IntArray(lienzo * lienzo)
        lienzoBmp.getPixels(px, 0, lienzo, 0, 0, lienzo, lienzo)
        lienzoBmp.recycle()

        val tipo = entrada.dataType()
        val q = entrada.quantizationParams()
        val bytesPorValor = if (tipo == DataType.FLOAT32) 4 else 1
        val buf = ByteBuffer.allocateDirect(px.size * 3 * bytesPorValor).order(ByteOrder.nativeOrder())
        fun poner(v: Float) {
            if (tipo == DataType.FLOAT32) {
                buf.putFloat(v)
            } else {
                val cuant = Math.round(v / q.scale + q.zeroPoint)
                buf.put((if (tipo == DataType.INT8) cuant.coerceIn(-128, 127) else cuant.coerceIn(0, 255)).toByte())
            }
        }
        if (nhwc) {
            for (p in px) {
                poner(Color.red(p) / 255f); poner(Color.green(p) / 255f); poner(Color.blue(p) / 255f)
            }
        } else {
            for (canal in 0 until 3) for (p in px) {
                poner(when (canal) { 0 -> Color.red(p); 1 -> Color.green(p); else -> Color.blue(p) } / 255f)
            }
        }
        buf.rewind()
        return buf
    }

    private fun leerTensor(indice: Int, b: ByteBuffer): Tensor {
        val t = interprete.getOutputTensor(indice)
        val forma = t.shape().drop(1).toIntArray()
        b.rewind()
        val n = t.numElements()
        val datos = when (t.dataType()) {
            DataType.FLOAT32 -> FloatArray(n).also { b.asFloatBuffer().get(it) }
            DataType.INT8 -> {
                val q = t.quantizationParams()
                FloatArray(n) { (b.get().toInt() - q.zeroPoint) * q.scale }
            }
            DataType.UINT8 -> {
                val q = t.quantizationParams()
                FloatArray(n) { ((b.get().toInt() and 0xFF) - q.zeroPoint) * q.scale }
            }
            else -> error("Tipo de salida no soportado: ${t.dataType()}")
        }
        return Tensor(forma, datos)
    }

    fun cerrar() = interprete.close()
}
