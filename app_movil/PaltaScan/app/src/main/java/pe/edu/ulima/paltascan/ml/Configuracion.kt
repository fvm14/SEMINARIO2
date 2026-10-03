package pe.edu.ulima.paltascan.ml

/** Como obtiene cada modelo las mascaras y la madurez. */
enum class TipoModelo {
    /** YOLOv8s-seg con cabezal de madurez: una sola red (el propuesto). */
    MULTITAREA,

    /** U-Net ResNet34: mascaras semanticas de palta y defecto + madurez, una sola red. */
    UNET,

    /** YOLOv8s-seg sin madurez + clasificador ResNet-34 de madurez. */
    DOS_REDES,
}

/**
 * Un modelo que la app puede usar. confDefecto es el umbral de defecto
 * calibrado en validacion para ese modelo (calibracion_defecto_val_resumen.csv
 * o eval_unet_multitarea.py --calibrar).
 */
data class ModeloApp(
    val id: String,
    val nombre: String,
    val tipo: TipoModelo,
    val archivo: String,
    val confDefecto: Float,
    val archivoClasificador: String? = null,
) {
    val archivos: List<String> get() = listOfNotNull(archivo, archivoClasificador)
}

object Configuracion {
    val MODELOS = listOf(
        ModeloApp("multitarea", "YOLOv8s-seg multitarea", TipoModelo.MULTITAREA, "palta_multitarea.tflite", 0.05f),
        ModeloApp("unet", "U-Net ResNet34 multitarea", TipoModelo.UNET, "unet_resnet34.tflite", 0.10f),
        ModeloApp(
            "dos_redes", "YOLOv8s-seg + ResNet-34", TipoModelo.DOS_REDES, "yolov8s_seg.tflite", 0.10f,
            archivoClasificador = "resnet34_madurez.tflite",
        ),
    )
    val MODELO_POR_DEFECTO = MODELOS.first()

    fun modelo(id: String?): ModeloApp = MODELOS.firstOrNull { it.id == id } ?: MODELO_POR_DEFECTO

    // Compatibilidad: el modelo propuesto
    val ARCHIVO_MODELO = MODELO_POR_DEFECTO.archivo

    const val CONF_PALTA = 0.35f

    // Umbral calibrado en validacion para yolov8s_mt_ronda1 (resultados/etapa2/*.json).
    val CONF_DEFECTO = MODELO_POR_DEFECTO.confDefecto

    /** U-Net: la palta se umbraliza en 0.5, como en eval_unet_multitarea.py. */
    const val UMBRAL_PALTA_UNET = 0.5f

    const val IOU_NMS = 0.7f
    const val MAX_DETECCIONES = 300

    // En pixeles de la foto original, igual que scripts/ocde.py.
    const val KERNEL_ROI_PX = 20

    const val HILOS_CPU = 4
}
