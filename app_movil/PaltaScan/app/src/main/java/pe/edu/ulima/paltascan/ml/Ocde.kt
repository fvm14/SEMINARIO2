package pe.edu.ulima.paltascan.ml

/** Port de scripts/ocde.py sobre mascaras planas (indice = y * ancho + x). */
object Ocde {
    // PENDIENTE: area de referencia sin fuente citable (ver progreso.md); mantener
    // igual que scripts/ocde.py hasta que se fije el valor definitivo.
    const val LIMITE_CAT_I = 0.094
    const val LIMITE_CAT_II = 0.141
    val CATEGORIAS = listOf("Categoría I", "Categoría II", "Rechazado")

    fun calcularRatio(fruto: BooleanArray, defecto: BooleanArray): Double {
        val pxFruto = fruto.count { it }
        if (pxFruto == 0) return 0.0
        return defecto.count { it }.toDouble() / pxFruto
    }

    fun clasificar(ratio: Double): Int = when {
        ratio <= LIMITE_CAT_I -> 0
        ratio <= LIMITE_CAT_II -> 1
        else -> 2
    }

    /** Mayor region 8-conexa. */
    fun mayorComponente(m: BooleanArray, ancho: Int, alto: Int): BooleanArray {
        val etiqueta = IntArray(m.size)
        val pila = IntArray(m.size)
        var mejor = 0
        var mejorArea = 0
        var siguiente = 0
        for (inicio in m.indices) {
            if (!m[inicio] || etiqueta[inicio] != 0) continue
            siguiente++
            var area = 0
            var tope = 0
            pila[tope++] = inicio
            etiqueta[inicio] = siguiente
            while (tope > 0) {
                val p = pila[--tope]
                area++
                val px = p % ancho
                val py = p / ancho
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = px + dx
                    val ny = py + dy
                    if (nx < 0 || ny < 0 || nx >= ancho || ny >= alto) continue
                    val q = ny * ancho + nx
                    if (m[q] && etiqueta[q] == 0) {
                        etiqueta[q] = siguiente
                        pila[tope++] = q
                    }
                }
            }
            if (area > mejorArea) {
                mejorArea = area
                mejor = siguiente
            }
        }
        return BooleanArray(m.size) { mejor != 0 && etiqueta[it] == mejor }
    }

    /**
     * Rellena los huecos: todo fondo no alcanzable desde el borde (4-conexo)
     * pasa a ser fruto, equivalente a dibujar relleno el contorno externo.
     */
    fun rellenarHuecos(m: BooleanArray, ancho: Int, alto: Int): BooleanArray {
        val exterior = BooleanArray(m.size)
        val pila = IntArray(m.size)
        var tope = 0
        fun sembrar(p: Int) {
            if (!m[p] && !exterior[p]) {
                exterior[p] = true
                pila[tope++] = p
            }
        }
        for (x in 0 until ancho) {
            sembrar(x)
            sembrar((alto - 1) * ancho + x)
        }
        for (y in 0 until alto) {
            sembrar(y * ancho)
            sembrar(y * ancho + ancho - 1)
        }
        while (tope > 0) {
            val p = pila[--tope]
            val px = p % ancho
            val py = p / ancho
            if (px > 0) sembrar(p - 1)
            if (px < ancho - 1) sembrar(p + 1)
            if (py > 0) sembrar(p - ancho)
            if (py < alto - 1) sembrar(p + ancho)
        }
        return BooleanArray(m.size) { !exterior[it] }
    }

    /** Region completa del fruto: mayor componente de (palta U defecto) sin huecos. */
    fun frutoCompleto(palta: BooleanArray, defecto: BooleanArray, ancho: Int, alto: Int): BooleanArray {
        val union = BooleanArray(palta.size) { palta[it] || defecto[it] }
        return rellenarHuecos(mayorComponente(union, ancho, alto), ancho, alto)
    }

    /** cv2.dilate con kernel cuadrado k x k (ancla en k / 2), separable. */
    fun dilatar(m: BooleanArray, ancho: Int, alto: Int, k: Int): BooleanArray {
        if (k <= 1) return m.copyOf()
        val antes = k / 2
        val despues = k - 1 - antes
        val horizontal = BooleanArray(m.size)
        for (y in 0 until alto) {
            val fila = y * ancho
            for (x in 0 until ancho) {
                if (!m[fila + x]) continue
                for (d in maxOf(0, x - despues)..minOf(ancho - 1, x + antes)) horizontal[fila + d] = true
            }
        }
        val salida = BooleanArray(m.size)
        for (x in 0 until ancho) {
            for (y in 0 until alto) {
                if (!horizontal[y * ancho + x]) continue
                for (d in maxOf(0, y - despues)..minOf(alto - 1, y + antes)) salida[d * ancho + x] = true
            }
        }
        return salida
    }

    fun filtrarDefectoPorRoi(fruto: BooleanArray, defecto: BooleanArray, ancho: Int, alto: Int, kernel: Int): BooleanArray {
        val roi = dilatar(fruto, ancho, alto, kernel)
        return BooleanArray(defecto.size) { defecto[it] && roi[it] }
    }
}
