package pe.edu.ulima.paltascan.ml

object Configuracion {
    const val ARCHIVO_MODELO = "palta_multitarea.tflite"

    const val CONF_PALTA = 0.35f

    // Umbral calibrado en validacion para yolov8s_mt_ronda1 (resultados/etapa2/*.json).
    // Si se cambia de modelo, usar el valor de su calibracion_defecto_val_resumen.csv.
    const val CONF_DEFECTO = 0.05f

    const val IOU_NMS = 0.7f
    const val MAX_DETECCIONES = 300

    // En pixeles de la foto original, igual que scripts/ocde.py.
    const val KERNEL_ROI_PX = 20

    const val HILOS_CPU = 4
}
