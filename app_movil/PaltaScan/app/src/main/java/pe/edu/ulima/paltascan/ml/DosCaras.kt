package pe.edu.ulima.paltascan.ml

/**
 * Evaluacion del fruto completo con dos fotos (lado A y lado B), igual que
 * scripts/dos_caras_evaluar.py.
 *
 * Cada foto muestra casi la mitad del fruto y su ratio defecto/fruto se
 * normaliza por el area proyectada de referencia (42.4 cm2). El defecto total
 * del fruto es (ratioA + ratioB) x 42.4 cm2, y compararlo con los 4 y 6 cm2 de
 * la norma equivale a aplicar los limites de Ocde a ratioA + ratioB. La
 * madurez se decide con el promedio de las probabilidades de las dos fotos.
 */
object DosCaras {

    class Combinado(val ratioA: Double, val ratioB: Double, val probabilidades: FloatArray) {
        val ratio: Double get() = ratioA + ratioB
        val categoria: Int get() = Ocde.clasificar(ratio)
        val madurez: Int get() = probabilidades.indices.maxBy { probabilidades[it] } + 1
        val probMadurez: Float get() = probabilidades.max()
    }

    /**
     * Avisos del fruto completo: los de foto oscura o borrosa de cualquiera de
     * los dos lados, y "madurez dudosa" segun la confianza de la madurez
     * combinada (la que se muestra), no la de cada foto por separado.
     */
    fun avisosCombinados(avisosA: List<String>, avisosB: List<String>, probMadurez: Float): List<String> {
        val baja = ValidacionCaptura.Problema.BAJA_CONFIANZA.name
        val deFoto = (avisosA + avisosB).filter { it != baja }.distinct()
        return if (probMadurez < ValidacionCaptura.CONFIANZA_MINIMA) deFoto + baja else deFoto
    }

    fun combinar(ratioA: Double, probsA: FloatArray, ratioB: Double, probsB: FloatArray): Combinado {
        require(probsA.size == probsB.size) { "Las dos fotos deben tener las mismas clases de madurez" }
        return Combinado(ratioA, ratioB, FloatArray(probsA.size) { (probsA[it] + probsB[it]) / 2f })
    }
}
