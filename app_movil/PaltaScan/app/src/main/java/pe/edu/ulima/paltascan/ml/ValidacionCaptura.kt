package pe.edu.ulima.paltascan.ml

/**
 * Revisa si la foto sirve para el analisis, con lo que el modelo ya entrega
 * (mascaras) y dos medidas simples de la imagen (brillo y nitidez).
 *
 * Bloqueantes (no se guarda el analisis): sin palta, mas de una palta, palta
 * cortada en el borde. Avisos (se guarda, pero se sugiere repetir): imagen
 * oscura, borrosa o madurez con baja confianza.
 *
 * Umbrales de imagen calibrados con las 108 fotos de prueba (caja de luz):
 * luminancia media >= 184 y varianza del Laplaciano >= 117 a 256 px, asi que
 * los limites de abajo no marcan ninguna foto valida del dataset.
 */
object ValidacionCaptura {

    enum class Problema(val bloqueante: Boolean) {
        SIN_PALTA(true),
        VARIAS_PALTAS(true),
        PALTA_CORTADA(true),
        OSCURA(false),
        BORROSA(false),
        BAJA_CONFIANZA(false),
    }

    const val LUMINANCIA_MINIMA = 70.0
    const val NITIDEZ_MINIMA = 40.0
    const val CONFIANZA_MINIMA = 0.6f
    const val LADO_CALIDAD = 256

    // Un fruto aparte cuenta si su area es al menos este fragmento del mayor.
    private const val FRACCION_FRUTO_EXTRA = 0.2

    // Pixeles del fruto sobre el borde, como fraccion del lado menor.
    private const val FRACCION_BORDE = 0.02

    fun revisar(
        fruto: BooleanArray, ancho: Int, alto: Int, nFrutos: Int,
        luminancia: Double, nitidez: Double, probMadurez: Float,
    ): List<Problema> = buildList {
        if (fruto.none { it }) {
            add(Problema.SIN_PALTA)
            return@buildList
        }
        if (nFrutos > 1) add(Problema.VARIAS_PALTAS)
        if (tocaBorde(fruto, ancho, alto)) add(Problema.PALTA_CORTADA)
        if (luminancia < LUMINANCIA_MINIMA) add(Problema.OSCURA)
        if (nitidez < NITIDEZ_MINIMA) add(Problema.BORROSA)
        if (probMadurez < CONFIANZA_MINIMA) add(Problema.BAJA_CONFIANZA)
    }

    fun tocaBorde(m: BooleanArray, ancho: Int, alto: Int): Boolean {
        var n = 0
        for (x in 0 until ancho) {
            if (m[x]) n++
            if (m[(alto - 1) * ancho + x]) n++
        }
        for (y in 1 until alto - 1) {
            if (m[y * ancho]) n++
            if (m[y * ancho + ancho - 1]) n++
        }
        return n >= maxOf(1.0, FRACCION_BORDE * minOf(ancho, alto))
    }

    /** Frutos separados (componentes 8-conexas) con area >= 20% del mayor. */
    fun contarFrutos(m: BooleanArray, ancho: Int, alto: Int): Int {
        val visto = BooleanArray(m.size)
        val pila = IntArray(m.size)
        val areas = ArrayList<Int>()
        for (inicio in m.indices) {
            if (!m[inicio] || visto[inicio]) continue
            var tope = 0
            pila[tope++] = inicio
            visto[inicio] = true
            var area = 0
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
                    if (m[q] && !visto[q]) {
                        visto[q] = true
                        pila[tope++] = q
                    }
                }
            }
            areas.add(area)
        }
        val mayor = areas.maxOrNull() ?: return 0
        return areas.count { it >= FRACCION_FRUTO_EXTRA * mayor }
    }

    /** Luminancia media (0-255) de una imagen en grises. */
    fun luminancia(gris: IntArray): Double = if (gris.isEmpty()) 0.0 else gris.average()

    /** Varianza del Laplaciano de 4 vecinos: baja = imagen borrosa. */
    fun nitidez(gris: IntArray, ancho: Int, alto: Int): Double {
        if (ancho < 3 || alto < 3) return 0.0
        var suma = 0.0
        var suma2 = 0.0
        var n = 0
        for (y in 1 until alto - 1) for (x in 1 until ancho - 1) {
            val i = y * ancho + x
            val l = (gris[i - ancho] + gris[i + ancho] + gris[i - 1] + gris[i + 1] - 4 * gris[i]).toDouble()
            suma += l
            suma2 += l * l
            n++
        }
        val media = suma / n
        return suma2 / n - media * media
    }
}
