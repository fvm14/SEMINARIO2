# Progreso del proyecto: palta-multitarea (Seminario de Investigación II)

Detección de defectos y clasificación de madurez de palta Hass con un único modelo multitarea, pensado para ejecutarse en un dispositivo móvil. Este documento resume lo realizado hasta el 27 de septiembre de 2026: el punto de partida, la arquitectura del modelo, las decisiones tomadas con su justificación, los resultados de las Etapas 1 y 2, y lo que queda pendiente.

## 1. Punto de partida

En Seminario I el pipeline se construyó en Roboflow: se anotaron 1,243 imágenes con polígonos de palta y defecto (con apoyo de SAM2), se entrenó un YOLOv8s-seg en Colab y se aplicó un motor OCDE que clasifica cada fruto en Categoría I, Categoría II o Rechazado según la proporción de superficie defectuosa. La madurez no formaba parte del modelo.

Para Seminario II se decidió reconstruir todo en Python, sin Roboflow, y convertirlo en un modelo multitarea que en una sola pasada segmente la palta y sus defectos y además clasifique la madurez en cinco niveles. La madurez se incorpora como un parche que debió existir en Seminario I, no como un paso nuevo del flujo de Seminario II.

Referencia de Seminario I (YOLOv8s-seg, split de Roboflow): P 0.801, R 0.763, F1 0.781, mAP50 0.785, mAP50-95 0.557. Estas cifras se obtuvieron con un split por imagen que filtraba frutos entre subconjuntos (ver sección 3), por lo que no son directamente comparables con las nuevas.

## 2. Arquitectura del modelo

La base es YOLOv8s-seg de Ultralytics (versión fijada en 8.4.160), inicializado con pesos COCO, al que se agregó un cabezal de madurez. El modelo recibe una imagen RGB de 800×800 px (letterbox con relleno gris 114) y produce tres resultados a la vez.

**Backbone (capas 0 a 9).** Red CSPDarknet con bloques C2f que extrae características a tres escalas: P3 (stride 8), P4 (stride 16) y P5 (stride 32). La capa 9 es un SPPF (Spatial Pyramid Pooling Fast) con 512 canales, que resume el contexto global de la imagen.

**Neck (capas 10 a 21).** Estructura PAN-FPN que combina las tres escalas en ambos sentidos (de arriba hacia abajo y de abajo hacia arriba) para que cada nivel tenga información semántica y espacial.

**Cabezal de segmentación (capa 22, Segment).** Trabaja sobre P3, P4 y P5. A 800 px esto equivale a grillas de 100×100, 50×50 y 25×25, es decir 13,125 anclas. Para cada ancla predice:

- la caja, como distribución discreta de distancias a los cuatro bordes (DFL, 16 valores por lado);
- la confianza de las dos clases, palta y defecto;
- 32 coeficientes de máscara.

Un módulo Proto genera, a partir de P3, 32 prototipos de máscara de 200×200. La máscara de cada instancia es sigmoid(coeficientes × prototipos), recortada a su caja.

**Cabezal de madurez (nuevo).** Se conecta a la salida del SPPF (capa 9) y reutiliza el backbone compartido: AdaptiveAvgPool → Linear(512, 256) → SiLU → Dropout(0.2) → Linear(256, 5). Entrega cinco logits, uno por nivel de madurez. Se eligió el SPPF porque es el punto con mayor contexto global antes de que el neck especialice las características para localización; la madurez es una propiedad de todo el fruto (color y textura de la cáscara), no de una región.

**Implementación.** La clase `YoloMultitarea` hereda de `SegmentationModel` y sobrescribe `_predict_once` para capturar la salida de la capa 9, pasarla por el cabezal y guardar los logits en `madurez_logits`. `__getstate__` excluye esos logits al guardar checkpoints. Todo está en `scripts/yolo_multitarea.py`.

**Función de pérdida.** Pérdida estándar de YOLO (caja, segmentación, clase y DFL) más `peso_madurez × CrossEntropy(madurez)`, con peso 2. La etiqueta de madurez se lee del nombre del archivo (por ejemplo `T10_d03_458_b_1.jpg`, donde el último número es el nivel).

**Selección del mejor checkpoint.** Fitness = 0.5 × (0.1 mAP50 + 0.9 mAP50-95 de máscaras) + 0.5 × F1 macro de madurez, para que ninguna de las dos tareas domine la elección.

**Post-proceso (común a Python y a la app).**

1. Umbral por clase: palta 0.35; defecto calibrado en validación.
2. NMS por clase (IoU 0.7, máximo 300 detecciones).
3. Construcción de máscaras y deshacer el letterbox.
4. `fruto_completo`: mayor componente conexo de palta ∪ defecto, con huecos rellenos.
5. Filtro ROI: se descartan defectos fuera del fruto dilatado 20 px.
6. Ratio = área de defecto / área de fruto, clasificado con los límites OCDE: 0.094 para Categoría I y 0.141 para Categoría II (4 y 6 cm² sobre 42.4 cm² de superficie media).
7. Madurez = argmax de las probabilidades.

La especificación de referencia está en NumPy en `scripts/inferencia_movil.py`, y es la que se portará a la app.

## 3. Datos

**Dataset original.** Hass Avocado Ripening Photographic Dataset: 14,710 fotografías de 478 paltas tomadas en distintos días y por ambos lados, con la madurez (1 a 5) en el nombre de archivo. De ellas, 1,243 (415 frutos) tienen anotación de segmentación de Seminario I.

**Split por fruto.** La misma palta aparece en varias fotos. El split de Roboflow era por imagen, y 198 de los 415 frutos anotados aparecían en más de un subconjunto; esa fuga probablemente inflaba las métricas de Seminario I. Se rehízo con StratifiedGroupKFold (10 pliegues asignados 7/2/1, semilla 42), agrupando por fruto y estratificando por madurez. En el dataset completo quedan 340 frutos en train, 94 en val y 44 en test. Los frutos nuevos (sin anotar) se asignaron con el mismo procedimiento sin mover los ya asignados. Archivos: `data/splits/split_manifest.csv` y `data/splits/manifest_completo.csv`.

**Auditoría de anotaciones.** Se revisaron las anotaciones de SAM2 y se excluyeron 51 imágenes (4.1%):

- 24 con polígono de palta diminuto o con área de defecto mayor que la palta;
- 27 cuyo defecto era un rectángulo de 4 vértices, que era el respaldo de SAM2 cuando fallaba.

El dataset limpio queda en 852 imágenes de train, 232 de val y 108 de test (39 frutos). Script: `scripts/auditar_anotaciones.py`; lista en `data/splits/anotaciones_excluidas.csv`.

## 4. Entrenamiento

Configuración común (hiperparámetros de Seminario I salvo lo indicado): imgsz 800, 100 épocas máximo, batch 8, patience 20, semilla 42, optimizador automático (resolvió a AdamW con lr ≈ 0.00167), `overlap_mask=False`, peso de madurez 2. Aumentos: rotación 15°, traslación 0.1, escala 0.5, volteos horizontal y vertical 0.5, hsv_h 0.01, hsv_s 0.4, hsv_v 0.3. Sin mosaic, mixup, cutmix ni copy-paste. Se entrenó en la laptop (RTX A5000 Laptop).

Modelos entrenados, en orden:

| Modelo | Cambio | Resultado |
|---|---|---|
| `yolov8s_multitarea(-2)` | primera versión | detectó el problema de huecos en la máscara de palta |
| `yolov8s_mt_limpio` | sin overlap, datos auditados, peso madurez 1 | base limpia; generó las pseudoetiquetas |
| `yolov8s_mt_peso2` | peso madurez 2 | elegido sobre peso 1 en validación |
| `yolov8s_mt_final` | peso 2, datos limpios, 852 imágenes | modelo base de la Etapa 1 |
| **`yolov8s_mt_ronda1`** | **+4,623 pseudoetiquetas** | **mejor modelo actual** |

Existe además una alternativa con ResNet34 archivada en `scripts/alternativa_resnet34/`, que se descartó al optar por YOLOv8s-seg con cabezal de madurez.

## 5. Protocolo de evaluación

- **Umbral de defecto calibrado en validación**, maximizando la IoU de defecto por píxel; nunca se ajusta sobre test.
- **Métricas de segmentación de Ultralytics:** P, R, mAP50 y mAP50-95 de máscaras.
- **Métricas por imagen:** IoU por píxel de palta y defecto, accuracy y F1 macro de madurez, madurez con tolerancia de ±1 nivel, acierto OCDE, recall de Rechazado, MAE del ratio y latencia.
- **Intervalos de confianza al 95% por bootstrap remuestreando frutos**, no imágenes, porque las imágenes de un mismo fruto no son independientes. Para comparar modelos se usa bootstrap pareado.
- **Visualización:** paneles ORIGINAL | ANOTACIÓN REAL | MODELO y un visor HTML con filtros (`scripts/visor.py`).

## 6. Etapa 1: ampliación del dataset con pseudo-etiquetado

**Objetivo.** Usar las 14,710 fotos y no solo las 1,243 anotadas. Revisar a mano miles de imágenes no era viable, así que se reemplazó la revisión manual por una regla de aceptación automática calibrada en validación.

**Procedimiento.**

1. Con `yolov8s_mt_limpio` se pseudo-etiquetaron las 9,892 fotos de train sin anotación (confianza palta 0.35, defecto 0.15).
2. Regla de aceptación (`scripts/calibrar_regla_pseudo.py`): se acepta una imagen si la fracción de área de defecto con confianza dudosa es ≤ 50%, no hay defectos rectangulares y la palta se detecta con confianza. En validación, las pseudoetiquetas aceptadas tuvieron precisión 0.85 y recall 0.49: son correctas pero incompletas.
3. Resultado: 4,623 aceptadas (47%) y 5,269 marcadas para revisar. Se guardó una muestra de control del 4% para estimar el error.
4. Dataset ampliado: train 852 + 4,623 = 5,475 imágenes; val y test intactos.
5. Reentrenamiento (Ronda 1): mismo modelo e hiperparámetros, 77 épocas, unas 2.4 h, mejor época la 57.
6. Recalibración del umbral de defecto: las pseudoetiquetas incompletas bajan la confianza del modelo en defectos, y el umbral óptimo pasó de 0.15 a 0.05. Este paso es obligatorio después de cada ronda.

**Resultados en test (108 imágenes, 39 frutos).**

| Métrica | Modelo base (umbral 0.15) | Ronda 1 (umbral 0.05) |
|---|---|---|
| Máscaras P / R | 0.829 / 0.734 | 0.794 / 0.771 |
| Máscaras F1 / mAP50 / mAP50-95 | 0.778 / 0.783 / 0.488 | 0.783 / 0.780 / 0.497 |
| IoU píxel palta / defecto | 0.996 / 0.420 | 0.994 / 0.397 |
| Madurez accuracy | 63.9% | **74.1%** [67.3, 80.9] |
| Madurez F1 macro | 0.633 | **0.737** [0.664, 0.804] |
| Madurez ±1 nivel | 100% | 100% |
| Acierto OCDE | 77.8% | 77.8% |
| Recall Rechazado | 64.3% | 60.7% |
| MAE ratio | 0.072 | 0.075 |

En el bootstrap pareado, la mejora en F1 de madurez es de +0.105, con IC [+0.005, +0.205], y es estadísticamente significativa. Las demás diferencias no lo son. El F1 de madurez por nivel en Ronda 1 es 0.889, 0.739, 0.711, 0.680 y 0.667; los niveles intermedios y avanzados son los más difíciles.

**Conclusión de la Etapa 1.** Ampliar el dataset mejora claramente la madurez, porque cada foto aporta su etiqueta real de madurez aunque sus defectos estén incompletos. La segmentación de defectos no mejora porque las pseudoetiquetas tienen recall bajo. El informe completo está en `resultados/Informe_Etapa1_Seminario2.pdf`.

## 7. Etapa 2: exportación y cuantización para móvil

**Exportación a ONNX** (`scripts/exportar_movil.py`). Se fusionan convolución y BatchNorm y se envuelve el modelo para que entregue cinco salidas separadas:

- `cajas` [1, 4, 13125], normalizadas entre 0 y 1;
- `puntajes` [1, 2, 13125];
- `coeficientes` [1, 32, 13125];
- `prototipos` [1, 32, 200, 200];
- `madurez` [1, 5], probabilidades con softmax.

La exportación usa entrada fija de 1×3×800×800, opset 17 y simplificación con onnxslim. El ONNX reproduce exactamente las métricas de PyTorch.

**Conversión a TFLite.** Con onnx2tf (de NCHW a NHWC, luego SavedModel) se generaron cuatro variantes por cuantización post-entrenamiento:

- **FP32:** conversión directa, de referencia.
- **FP16:** pesos en media precisión.
- **Dynamic range:** pesos en INT8 por canal y activaciones cuantizadas en tiempo de ejecución; no necesita calibración.
- **INT8 completo:** pesos y activaciones en INT8, con rangos estimados sobre 16 imágenes de calibración; entrada y salida quedan en float.

**Problema encontrado y solución.** En la primera versión, el INT8 daba IoU 0 en palta y defecto. La causa es que el cabezal de Ultralytics concatena por dentro las cajas en píxeles (0 a 800) con las confianzas (0 a 1) en un solo tensor. La cuantización por tensor usa una única escala, de unos 3.3 por paso, y todas las confianzas quedan en cero. Se diagnosticó leyendo los tensores internos del modelo cuantizado. La solución fue reescribir la etapa final del cabezal para emitir cajas normalizadas, confianzas y coeficientes por separado, sin esa concatenación.

**Resultados en test** (108 imágenes, umbral de defecto 0.05, CPU con 4 hilos del entorno de pruebas; la latencia en celular está por medir):

| Versión | Tamaño | IoU defecto | Madurez | OCDE | Recall rechazado | Latencia |
|---|---|---|---|---|---|---|
| ONNX FP32 | 48 MB | 0.397 | 74.1% | 77.8% | 60.7% | 637 ms |
| TFLite FP16 | 24 MB | 0.397 | 74.1% | 77.8% | 60.7% | 430 ms |
| **TFLite dynamic range** | **12 MB** | **0.399** | **73.1%** | **79.6%** | **60.7%** | **175 ms** |
| TFLite INT8 | 12 MB | 0.402 | 65.7% | 75.9% | 39.3% | 125 ms |

**Decisión.** Dynamic range es el modelo principal para la app en CPU: un cuarto del tamaño, sin pérdida relevante y 3.6 veces más rápido que FP32. FP16 queda para probar con el delegado GPU del celular. El INT8 completo pierde 8 puntos de madurez y la detección de rechazados cae de 61% a 39%, porque las confianzas bajas de defecto y el cabezal de clasificación son los más sensibles al redondeo. Se documenta como hallazgo: el costo de la cuantización completa en un modelo multitarea.

Los modelos están en `modelos_movil/` y las métricas en `resultados/etapa2/`.

## 8. Decisiones principales y su justificación

| Decisión | Motivo |
|---|---|
| YOLOv8s-seg + cabezal de madurez (un solo modelo) | Una sola pasada en el celular; el backbone compartido evita duplicar cómputo |
| Cabezal de madurez en el SPPF | Máximo contexto global; la madurez es una propiedad de todo el fruto |
| Split por fruto con StratifiedGroupKFold | Eliminar la fuga de Seminario I (198 frutos repetidos entre splits) |
| Excluir 51 anotaciones defectuosas | Fallas de SAM2 que enseñaban al modelo defectos rectangulares o paltas diminutas |
| Sin mosaic, mixup ni copy-paste | Mezclarían paltas de distinta madurez en una imagen con una sola etiqueta |
| Aumentos de color suaves (hsv_s 0.4, hsv_v 0.3) | La madurez Hass se ve en el color de la cáscara; aumentos fuertes borran la señal |
| `overlap_mask=False` y `fruto_completo` | La máscara de palta tenía huecos donde había defectos, y el filtro ROI borraba defectos grandes (el acierto OCDE subió de 65% a 75%) |
| Peso de madurez 2 | Equilibra ambas tareas; elegido en validación |
| Umbral por clase calibrado en validación | Los defectos tienen confianzas más bajas que la palta; nunca se ajusta en test |
| Regla automática de aceptación de pseudoetiquetas | Revisar miles de imágenes a mano no era viable; la regla se calibró en validación |
| Bootstrap por fruto | Las imágenes de un mismo fruto no son independientes |
| Dynamic range como modelo móvil | Mejor equilibrio entre tamaño, precisión y latencia |
| App nativa Android (Kotlin, LiteRT, CameraX) | Acceso directo a delegados GPU y NNAPI y menor latencia |

## 9. Repositorio y entregables

- **Repositorio:** github.com/fvm14/SEMINARIO2. Contiene código, splits, anotaciones de Seminario I (`data/raw/labels_seg`), métricas, curvas e informe.
- **Fuera del repo por tamaño** (definido en `.gitignore`): dataset crudo, imágenes, pseudoetiquetas y modelos.
- **Release `v0.2-etapa2`:** incluye `ronda1_best.pt`, `limpio_best.pt`, el ONNX, las tres variantes TFLite, `imagenes_anotadas.zip` y `pseudoetiquetas.zip` (preparados en `Documents/release_etapa2`).
- **Dataset crudo (2.5 GB):** se comparte aparte, por la fuente original o por Drive. Es necesario para reproducir la Ronda 1.
- **Otros productos:**
  - Informe de la Etapa 1 en PDF
  - Diagrama de arquitectura (`resultados/arquitectura_yolov8s_multitarea.png`)
  - Historias de usuario (productor o acopiador, operario de packing)
  - Prompt de diseño para las pantallas de la app (`frontend/PaltaScan — Pantallas.pdf`)

## 10. Pendientes

**Cierre de la Etapa 2**
- Exportar y evaluar a 640 px.
- Regenerar el conjunto de calibración desde train, estratificado por madurez, con 100 a 200 imágenes.
- Actualizar el README, que aún describe `yolov8s_mt_peso2` como modelo final.

**Dominio real**
- Tomar de 50 a 100 fotos con celular en condiciones de campo y de packing, y medir cuánto cae el desempeño frente a las fotos de laboratorio. Es el riesgo principal del proyecto.

**Cierre de la Etapa 1**
- Respuesta de Dylann sobre cómo se trataron las lenticelas en Seminario I.
- Guía de anotación.
- Corrección dirigida de unas 200 imágenes.
- Posible Ronda 2.
- Revisión opcional de la muestra de control para reportar la tasa de error de las pseudoetiquetas.

**Etapa 4: app**
- Portar el post-proceso de `inferencia_movil.py` a Kotlin.
- Integrar LiteRT y CameraX.
- Medir la latencia real en celular, que es la cifra que va a la tesis.

**Paper**
- Redacción con las tablas y los intervalos de confianza de este documento.
