# PaltaScan — app Android

App Android nativa (Kotlin) que evalua una palta desde una foto **en el propio
celular, sin internet**: segmenta la palta y sus defectos, clasifica la
madurez (1-5) y asigna la categoria OCDE. Guarda un historial por lote que se
exporta a CSV y PDF. Detalle de todas las funciones en `FUNCIONALIDADES.md`.

## Estructura

```
PaltaScan/app/src/main/java/pe/edu/ulima/paltascan/
  ml/            Nucleo del modelo, port de scripts/inferencia_movil.py y scripts/ocde.py
    Configuracion.kt     modelos disponibles y umbrales (palta 0.35, defecto por modelo, IoU 0.7, ROI 20 px)
    Letterbox.kt         geometria del letterbox (igual que en Python)
    LectorSalidas.kt     identifica las salidas del .tflite por su forma
    Postproceso.kt       YOLO: confianza, NMS por clase, mascara = sigmoide(coef . prototipos);
                         U-Net: recorte y umbral de las mascaras semanticas
    Ocde.kt              fruto completo, filtro ROI, ratio defecto/fruto, categoria
    ValidacionCaptura.kt sin palta, varias, cortada, oscura, borrosa, baja confianza
    Analizador.kt        pipeline completo con LiteRT y tiempos por etapa
  datos/         SQLite (lotes e historial), CSV, PDF y utilidades de imagen
  ui/            Pantalla principal e historial
PaltaScan/app/src/test/          pruebas de la logica en la PC (sin celular ni modelo)
PaltaScan/app/src/androidTest/   ConcordanciaTest: analiza las fotos de prueba en el celular
```

## Modelos

La app trae el modelo propuesto y, para la comparacion de modelos, puede
incluir los otros dos. Cada uno va en `PaltaScan/app/src/main/assets/` con
este nombre (estan en `.gitignore`: no se suben al repo):

| Modelo | Archivo(s) en assets | Umbral de defecto |
|---|---|---|
| YOLOv8s-seg multitarea (propuesto) | `palta_multitarea.tflite` | 0.05 |
| U-Net ResNet34 multitarea | `unet_resnet34.tflite` | 0.10 |
| YOLOv8s-seg + ResNet-34 (dos redes) | `yolov8s_seg.tflite` y `resnet34_madurez.tflite` | 0.10 |

Si hay mas de un modelo, la pantalla principal muestra la fila "Modelo" para
elegir con cual analizar; cada analisis guarda el modelo usado.

**Generar los modelos alternativos** (desde la raiz del proyecto):

```bash
# 1) ONNX (entorno del proyecto, con PyTorch)
python scripts/exportar_alternativos.py --unet resultados/comparacion/unet_resnet34/best.pt \
    --yolo resultados/yolo_multitarea/yolov8s_base_sem1/weights/best.pt \
    --resnet resultados/comparacion/resnet34_madurez/best.pt
# 2) TFLite float32, float16 y rango dinamico (entorno aparte con tensorflow y onnx2tf)
python scripts/onnx_a_tflite.py modelos_movil/unet_resnet34_800.onnx \
    modelos_movil/yolov8s_seg_800.onnx modelos_movil/resnet34_madurez_448.onnx
# 3) Evaluar en test lo que se pierde al cuantizar (y dejar la referencia para la app)
python scripts/eval_alternativos_movil.py --tipo unet --conf-defecto 0.10 \
    --modelo modelos_movil/unet_resnet34_800_dynamic_range_quant.tflite --salida resultados/etapa3_movil/unet_dynamic_range
python scripts/eval_alternativos_movil.py --tipo dos_redes --conf-defecto 0.10 \
    --modelo modelos_movil/yolov8s_seg_800_dynamic_range_quant.tflite \
    --clasificador modelos_movil/resnet34_madurez_448_dynamic_range_quant.tflite --salida resultados/etapa3_movil/dos_redes_dynamic_range
```

Luego copiar los `*_dynamic_range_quant.tflite` a assets con los nombres de la tabla.

## Compilar y probar

```bash
cd app_movil/PaltaScan
./gradlew testDebugUnitTest   # 23 pruebas de la logica
./gradlew installDebug        # instala en el celular conectado (MIUI pide aceptar)
```

## Verificar contra Python en el celular

1. Copiar las 108 fotos de prueba a la app:
   `adb push data/prueba_test/. /sdcard/Android/data/pe.edu.ulima.paltascan/files/prueba/`
2. Correr `./gradlew connectedDebugAndroidTest` (todos los modelos instalados;
   con `-Pandroid.testInstrumentationRunnerArguments.modelo=unet` solo uno).
3. Sacar los resultados: `adb exec-out run-as pe.edu.ulima.paltascan cat files/resultados_app_unet.json`
   (`resultados_app.json` para el propuesto).
4. Comparar: `python scripts/eval_alternativos_movil.py --comparar <referencia_app.json> <resultados_app_*.json>`
   o, para el propuesto, `scripts/referencia_app.py`.

La categoria OCDE y la madurez deben coincidir; el ratio puede diferir en
milesimas en fotos que no son cuadradas, porque la app trabaja a la resolucion
del letterbox.

## Diferencias con Python (conscientes)

- La foto se reduce a un lado maximo de 1600 px antes de analizarla, y las
  mascaras y el ratio se calculan en el espacio del letterbox (maximo
  800x800) en vez de a la resolucion original. El kernel ROI se escala en
  proporcion.
- Modo foto por foto (no video en vivo).

## Pendiente

- [x] Modelos alternativos cuantizados e instalados en assets.
- [x] Latencia en el celular de los tres modelos y concordancia con Python.
- [ ] Camara en vivo con medicion de FPS.
- [ ] Delegado GPU.
