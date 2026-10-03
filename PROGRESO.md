# Progreso del proyecto: palta-multitarea (Seminario de Investigación II)

Detección de defectos y clasificación de madurez de palta Hass con un único modelo multitarea, pensado para ejecutarse en un dispositivo móvil. Este documento resume lo realizado hasta el 2 de octubre de 2026: el punto de partida, la arquitectura del modelo, las decisiones tomadas con su justificación, los resultados de las Etapas 1 a 5 y lo que queda pendiente.

**Etapas:** 1) ampliación del dataset con pseudo-etiquetado; 2) cuantización para móvil; 3) validación de métricas (comparación con otros modelos); 4) prototipo (app Android); 5) integración del modelo y el motor OCDE en la app.

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

**Línea base de tarea única (modelo de Seminario I), en curso.** Se descargó el `best.pt` de defectos de Seminario I (Drive `SEMINARIO 1/MODELOS/Modelos/YOLOv8s/.../weights`, ahora `weights/s1_defectos_best.pt`) para compararlo con Ronda 1 en la actividad 1.3. Hallazgo: **no se puede evaluar directamente sobre nuestro val/test**, porque se entrenó con el split aleatorio de Roboflow sobre las mismas fotos. De nuestro test, 77 de 108 fotos y 37 de 39 frutos estuvieron en su entrenamiento; de val, 168 de 232 fotos y 72 de 79 frutos. Por eso en val da una IoU de defecto de 0.62 (umbral 0.20) frente a 0.47 de Ronda 1: la cifra está inflada. La lista de fotos de su train está en `resultados/linea_base_s1/s1_train_stems.txt`; el script de evaluación es `scripts/eval_linea_base_s1.py`. Evaluado igual sobre ese test contaminado, da mAP50 0.889, mAP50-95 0.658, IoU de defecto 0.548 y recall de Rechazado 75% (`resultados/linea_base_s1/prueba_contaminada/`, solo como evidencia de la fuga, no reportable).

Solución preparada para correr en Colab (o en la RTX A5000 del compañero): `notebooks/linea_base_s1_colab.ipynb` con `scripts/train_linea_base_s1.py`. Reentrena YOLOv8s-seg de tarea única sobre nuestro train de 852 imágenes en dos configuraciones: `replica_s1` (hiperparámetros exactos de Seminario I) y `ablacion` (los de `yolov8s_mt_final`, para aislar el efecto del cabezal de madurez). Luego calibra el umbral en val, evalúa en test y hace bootstrap pareado contra `mt_final` y `mt_ronda1`. Guarda todo en Drive (`SEMINARIO 1/SEMI2/linea_base_s1`) y se reanuda si Colab se corta. Requiere hacer push antes, porque Colab clona el repo.

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

## 8. Etapa 3: validación de métricas (comparación de modelos)

Nueva etapa, entre la cuantización y el prototipo. Compara el modelo propuesto con otros paradigmas de visión computacional para sustentar su elección con métricas. Los modelos alternativos se entrenan y evalúan, pero no se cuantizan ni se integran en la app. Realizada el 1 y 2 de octubre de 2026 (Fabrizio).

### 8.1 Decisiones

**No comparar versiones de YOLO.** En la primera fase ya se compararon YOLOv8s, YOLOv9 y YOLOv11s con resultados muy parejos, porque comparten paradigma. Repetirlo daría diferencias que, con 108 imágenes de prueba, no se distinguen del azar. Se decidió variar el paradigma y no la versión.

**Filtrar las alternativas de la literatura por su capacidad de resolver la tarea.** El sistema necesita el área del defecto para calcular la proporción y asignar la categoría OCDE, lo que exige segmentación. Quedaron descartados:

- Los detectores por cajas (Faster R-CNN, SSD, RetinaNet, RT-DETR), porque el área de una caja no es el área del defecto.
- Los clasificadores como modelo único (VGG, Inception, DenseNet), porque no delimitan defectos.
- Los modelos que requieren datos hiperespectrales o NIR, porque el dataset es RGB.

**Tres modelos, todos con las dos salidas (fallas y madurez).** Ninguna arquitectura trae ambas tareas de fábrica, así que en cada caso se agrega la madurez con un cabezal o con una segunda red:

| # | Modelo | Paradigma | Pregunta que responde |
|---|---|---|---|
| 1 | YOLOv8s-seg multitarea (propuesto) | Una red, segmentación de instancias en una etapa, cabezal de madurez | Es la propuesta |
| 2 | U-Net con ResNet34, multitarea | Una red, segmentación semántica, cabezal de madurez | ¿Instancias o segmentación semántica? |
| 3 | YOLOv8s-seg + ResNet-34 | Dos redes de tarea única | ¿Multitarea o redes especializadas? |

Mask R-CNN (instancias en dos etapas) se dejó fuera por costo: requiere código a medida y entre 4 y 6 horas de entrenamiento. Puede agregarse como cuarta fila sin rehacer lo anterior.

**Condiciones idénticas.** Los tres modelos usan la misma partición por fruto, las mismas 5,475 imágenes de entrenamiento (852 anotadas más 4,623 pseudoetiquetas), la misma validación (232) y prueba (108) con anotación humana, semilla 42, y el umbral de defecto calibrado en validación por separado para cada uno.

**Criterio de selección.** Se selecciona el modelo de menor tamaño y latencia entre aquellos cuyo acierto OCDE y exactitud de madurez no sean significativamente inferiores al mejor.

### 8.2 Modelos entrenados

| Modelo | Script | Configuración | Entrenamiento |
|---|---|---|---|
| YOLOv8s-seg multitarea | `train_yolo_multitarea.py` | 800 px, lote 8, peso de madurez 2, sin mosaic | Ronda 1 (ya existía): 77 épocas, mejor la 57 |
| YOLOv8s-seg sin madurez | `train_base_sem1.py` | Configuración de Seminario I: mosaic 1.0, hsv 0.015/0.7/0.4 | 97 épocas (parada temprana), mejor la 77, 3.2 h |
| U-Net ResNet34 multitarea | `unet_multitarea.py` | 800 px, lote 8, AdamW, peso de madurez 2, mismos aumentos que el multitarea | 57 épocas (parada temprana), mejor la 37, 2.8 h |
| ResNet-34 de madurez | `clasificador_madurez.py` | 448 px, lote 32, AdamW | 28 épocas (parada temprana), mejor la 18, unos 13 min |

Detalles de arquitectura:

- **U-Net ResNet34:** codificador ResNet34 preentrenado en ImageNet, decodificador U-Net con conexiones de salto y dos máscaras independientes (palta y defecto), más un cabezal de madurez sobre el último mapa del codificador. Pérdida: entropía cruzada binaria más Dice para la segmentación, y entropía cruzada para la madurez. 24.6 millones de parámetros.
- **ResNet-34 de madurez:** ResNet-34 preentrenada en ImageNet con salida de 5 niveles. 21.3 millones de parámetros.
- **Dos redes:** las fallas y la categoría OCDE provienen del YOLOv8s-seg sin madurez; la madurez, del clasificador. La latencia es la suma de ambas.

Scripts de evaluación: `eval_yolo_multitarea.py` (adaptado para aceptar modelos sin cabezal de madurez), `eval_unet_multitarea.py` (calibración y evaluación) y `clasificador_madurez.py --evaluar` (evalúa y combina con el segmentador). Todos generan `predicciones.csv` y `metricas.json` en el mismo formato, de modo que `bootstrap_ic.py` compara cualquier par de modelos.

### 8.3 Resultados en test

108 imágenes de 39 frutos. Entre corchetes, IC 95% por bootstrap agrupado por fruto (5,000 remuestreos).

| Métrica | YOLOv8s-seg multitarea (propuesto) | U-Net ResNet34 multitarea | Dos redes: YOLOv8s-seg + ResNet-34 |
|---|---|---|---|
| IoU de defecto (píxel) | 0.397 | 0.396 | 0.443 |
| Acierto OCDE | 77.8% [65.7, 87.9] | 65.7% [55.7, 75.7] | 82.4% [73.0, 90.9] |
| F1 macro OCDE | 0.654 | 0.467 | 0.749 |
| Recall de Rechazado | 60.7% | 46.4% | 53.6% |
| MAE del ratio | 0.075 | 0.067 | 0.065 |
| Exactitud de madurez | 74.1% [67.3, 81.0] | 73.1% [64.2, 81.7] | 77.8% [69.7, 85.3] |
| F1 macro de madurez | 0.737 | 0.735 | 0.772 |
| Parámetros | 11.9 M | 24.6 M | 33.1 M (11.8 + 21.3) |
| Tamaño FP32 | 48 MB | 98 MB | 132 MB |
| Latencia en CPU | 416 ms | 856 ms | 584 ms (422 + 162) |
| Umbral de defecto | 0.05 | 0.10 | 0.10 |

La latencia corresponde solo a la inferencia, medida en PyTorch sobre la CPU del entorno de pruebas, con entrada de 800 px (448 px el clasificador) y media de 15 pasadas. Sirve para comparar los modelos entre sí; no es la latencia del teléfono.

### 8.4 Diferencias frente al propuesto (bootstrap pareado)

| Métrica | U-Net menos propuesto | Dos redes menos propuesto |
|---|---|---|
| Acierto OCDE | -12.0 puntos [-20.3, -3.4], significativa | +4.6 puntos [-1.7, +11.1], no significativa |
| F1 macro OCDE | -0.187 [-0.287, -0.058], significativa | +0.095 [+0.008, +0.198], significativa |
| Recall de Rechazado | -14.3 puntos [-27.6, 0.0], no significativa | -7.1 puntos [-22.9, +7.1], no significativa |
| Exactitud de madurez | -0.9 puntos [-8.8, +6.3], no significativa | +3.7 puntos [-4.0, +11.7], no significativa |
| F1 macro de madurez | -0.002 [-0.081, +0.071], no significativa | +0.035 [-0.049, +0.117], no significativa |
| MAE del ratio | -0.009 [-0.034, +0.011], no significativa | -0.010 [-0.028, +0.005], no significativa |

### 8.5 Interpretación

**U-Net frente al propuesto.** Ambos empatan en IoU de defecto y en madurez, pero la U-Net acierta la categoría OCDE 12 puntos menos, con diferencia significativa, y requiere el doble de parámetros y de latencia. Queda descartada.

**Dos redes frente al propuesto.** Las dos redes obtienen los mejores valores en casi todas las métricas de desempeño. Las diferencias en acierto OCDE y en madurez no son significativas; la del F1 macro de OCDE sí lo es, a favor de las dos redes. El propuesto conserva mejor recall de Rechazado, sin significancia. A cambio, las dos redes ocupan 2.8 veces más y tardan 40% más.

**Conclusión.** La comparación sustenta la elección por equilibrio, no por superioridad. El modelo multitarea no es significativamente inferior al mejor en acierto OCDE ni en madurez, y es el de menor tamaño y latencia, por lo que cumple el criterio de selección. No corresponde afirmar que es el más exacto: cede una ventaja pequeña en exactitud a cambio de una ganancia grande en eficiencia, que es lo que pesa en un dispositivo móvil.

**Efecto de agregar la madurez.** La comparación entre el YOLOv8s-seg sin madurez y el multitarea indica que la segunda tarea tiene un costo pequeño en segmentación (IoU de defecto de 0.443 a 0.397; acierto OCDE de 82.4% a 77.8%, no significativo). Las dos configuraciones difieren además en los aumentos (mosaic y color), por lo que el efecto no puede atribuirse solo al cabezal.

### 8.6 Limitaciones

- Una sola corrida por modelo, con semilla 42. No se midió la variación entre semillas.
- El conjunto de prueba es pequeño (108 imágenes, 39 frutos), lo que da intervalos amplios.
- El umbral de la U-Net (0.10) coincidió con el mínimo del barrido original. Se agregó 0.05 al barrido; falta repetir la calibración para confirmar.
- El clasificador ResNet-34 usa entrada de 448 px, frente a 800 px de los modelos de segmentación.
- La latencia se midió en computadora y sin cuantizar. Solo el modelo propuesto tiene medición en teléfono.
- Todas las imágenes provienen de condiciones controladas de iluminación.

### 8.7 Archivos

| Qué | Dónde |
|---|---|
| Tabla comparativa | `resultados/comparacion/tabla_comparativa.md` |
| U-Net: pesos, curvas y evaluación | `resultados/comparacion/unet_resnet34/` |
| Clasificador: pesos y curvas | `resultados/comparacion/resnet34_madurez/` |
| Dos redes: predicciones y métricas combinadas | `resultados/comparacion/dos_redes/` |
| YOLOv8s-seg sin madurez | `resultados/yolo_multitarea/yolov8s_base_sem1/` |
| Modelo propuesto | `resultados/yolo_multitarea/yolov8s_mt_ronda1/` |
| Scripts nuevos | `scripts/unet_multitarea.py`, `scripts/eval_unet_multitarea.py`, `scripts/clasificador_madurez.py`, `scripts/train_base_sem1.py` |

### 8.8 Pendientes

1. Repetir la calibración de la U-Net con el barrido ampliado y reevaluar si cambia el umbral:
   `python scripts/eval_unet_multitarea.py --weights resultados\comparacion\unet_resnet34\best.pt --calibrar`
2. Opcional: agregar Mask R-CNN como cuarto modelo.
3. Opcional: reentrenar el multitarea con mosaic activado, para comprobar si recupera la diferencia de segmentación frente al modelo sin madurez.
4. Cuantizar los dos modelos alternativos y medir su latencia en el celular (sección 8.9).

### 8.9 Cuantización de los modelos alternativos y uso en la app (en curso)

Objetivo: medir en el teléfono, y no solo en la PC, cuánto tardan los tres modelos, con el mismo esquema de cuantización que el propuesto (rango dinámico).

**Hecho (código, probado con pesos de prueba):**
- `scripts/exportar_alternativos.py`: exporta a ONNX la U-Net (salidas `mascaras` [1, 2, 800, 800] con sigmoide y `madurez` [1, 5]), el YOLOv8s-seg sin madurez (mismas 4 salidas separadas que el multitarea, sin la madurez) y la ResNet-34 (`madurez` [1, 5], entrada de 448 px). La normalización de ImageNet va dentro del modelo, así la app prepara la entrada igual para todos (RGB 0-1 con letterbox). Compara cada ONNX con PyTorch (diferencia máxima ≤ 2e-4).
- `scripts/onnx_a_tflite.py`: ONNX → TFLite con onnx2tf (convertidor de TensorFlow, como en la Etapa 2) en float32, float16 y rango dinámico. Se ejecuta en un entorno aparte con tensorflow 2.20 y onnx2tf. El float32 reproduce al ONNX (diferencia ≤ 6e-6). Tamaños en rango dinámico: U-Net 24.7 MB, YOLOv8s-seg 12.2 MB, ResNet-34 21.4 MB.
- `scripts/eval_alternativos_movil.py`: evalúa los TFLite en test con el mismo protocolo que sus versiones en PyTorch (`metricas.json`, `predicciones.csv` para `bootstrap_ic.py` y `referencia_app.json` para comparar con la app) y compara la salida de la app con esa referencia (`--comparar`).
- App v0.4.0: soporta los tres modelos (sección 12).

**Falta:** los pesos `best.pt` de la U-Net, la ResNet-34 y el YOLOv8s-seg sin madurez (se entrenaron en otra máquina y no están en el repo); con ellos, convertir, evaluar la pérdida por cuantización, instalarlos en el celular y correr `ConcordanciaTest` con cada uno.

## 9. Decisiones principales y su justificación

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
| YOLOv8s-seg multitarea frente a U-Net y a dos redes | No es significativamente inferior al mejor en acierto OCDE ni en madurez, y es el de menor tamaño y latencia (sección 8) |
| Dynamic range como modelo móvil | Mejor equilibrio entre tamaño, precisión y latencia |
| App nativa Android (Kotlin, LiteRT, CameraX) | Acceso directo a delegados GPU y NNAPI y menor latencia |

## 10. Repositorio y entregables

- **Repositorio:** github.com/fvm14/SEMINARIO2. Contiene código, splits, anotaciones de Seminario I (`data/raw/labels_seg`), métricas, curvas e informe.
- **Fuera del repo por tamaño** (definido en `.gitignore`): dataset crudo, imágenes, pseudoetiquetas y modelos.
- **Release `v0.2-etapa2`:** incluye `ronda1_best.pt`, `limpio_best.pt`, el ONNX, las tres variantes TFLite, `imagenes_anotadas.zip` y `pseudoetiquetas.zip` (preparados en `Documents/release_etapa2`).
- **Dataset crudo (2.5 GB):** se comparte aparte, por la fuente original o por Drive. Es necesario para reproducir la Ronda 1.
- **Otros productos:**
  - Informe de la Etapa 1 en PDF
  - Diagrama de arquitectura (`resultados/arquitectura_yolov8s_multitarea.png`)
  - Historias de usuario (productor o acopiador, operario de packing)
  - Prompt de diseño para las pantallas de la app (`frontend/PaltaScan — Pantallas.pdf`)

## 11. Pendientes

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

**Etapa 4: app** (ver sección 12)
- ~~Portar el post-proceso de `inferencia_movil.py` a Kotlin.~~ Hecho: `app_movil/PaltaScan/`, con 13 pruebas unitarias.
- ~~Integrar LiteRT.~~ Hecho con TensorFlow Lite 2.16. En lugar de CameraX se usa la cámara del sistema, porque el análisis es foto por foto.
- Medir la latencia real en celular y correr la prueba de concordancia app vs. Python (`ConcordanciaTest`).

**Paper**
- Redacción con las tablas y los intervalos de confianza de este documento.

## 12. Etapas 4 y 5: prototipo e integración (Dylann)

**Etapa 4: prototipo (app Android `app_movil/PaltaScan/`).**
- Decisión: app Android nativa en Kotlin, con inferencia en el celular (TFLite dynamic range) y sin nube obligatoria.
- Hecho: port a Kotlin de `inferencia_movil.py` y `ocde.py` (letterbox, lectura de las 5 salidas, NMS por clase, máscaras, fruto completo, ROI, ratio, categoría OCDE) y tres pantallas:
  - **Inicio:** lote, cámara o galería.
  - **Resultado:** original vs. máscara, OCDE, madurez y ms por etapa.
  - **Historial por lote:** con exportación a CSV y guardado en SQLite.
- Pasan 13 pruebas unitarias y el APK compila. El modelo dynamic range ya está en los assets.
- Validado en Python (`scripts/referencia_app.py`) sobre las 108 fotos de prueba: 73.15% de madurez, igual que en `resultados/etapa2`. Resultado en `resultados/app/referencia_python_test.json`.
- Nuevo `scripts/probar_modelo_gui.py`: ventana tkinter para probar el modelo con fotos. Versión 2: lista desplegable con los modelos que encuentra (TFLite/ONNX, `.pt` multitarea y `.pt` de solo defectos, incluidas las líneas base cuando existan), cada uno con su umbral calibrado; paneles original / anotación real / modelo; por foto, OCDE y % de defecto frente al real, IoU, precisión y recall de defecto, madurez frente a la real y barras de probabilidad por nivel; métricas acumuladas y botón "Evaluar carpeta completa". Verificada sobre las 108 fotos de test: reproduce las cifras reportadas (PyTorch 74.1% madurez, IoU 0.397, OCDE 77.8%, Rechazado 60.7%; TFLite 73.1%).
- **Prueba en celular real (2026-09-28): Xiaomi 11 Lite 5G NE (Snapdragon 778G, Android 14).**
  - Error encontrado: TFLite 2.16.1 no podía cargar el modelo (`FULLY_CONNECTED` versión 12, generado por un conversor más nuevo). Se cambió la dependencia a LiteRT 1.4.2 (`com.google.ai.edge.litert:litert`, mismo API `org.tensorflow.lite`). LiteRT 2.x exige Kotlin 2, así que se descartó por ahora.
  - `ConcordanciaTest` en las 108 fotos de test, contra la referencia en Python (`resultados/app/resultados_app_xiaomi11lite.json` vs `referencia_python_test.json`): madurez idéntica en 108/108 (diferencia máxima de probabilidad 0.0045), máscara del fruto idéntica, categoría OCDE igual en 107/108 y ratio de defecto con mediana de diferencia 0. En 6% de las fotos el ratio difiere, con un máximo de +17 puntos: son defectos con confianza en el límite del umbral 0.05 que se activan por diferencias numéricas mínimas.
  - Latencia en el celular (CPU, LiteRT): preproceso 104 ms, inferencia 715 ms y postproceso 80 ms, unos **0.9 s por foto**. Es 4 veces más lenta que en la PC (175 ms). Se puede reducir probando el delegado GPU (con el modelo FP16) o más hilos.
  - En MIUI la instalación por USB exige aceptar un aviso por cada APK, y adb no puede leer `Android/data`; la prueba ahora guarda también una copia en el almacenamiento interno (`run-as ... cat files/resultados_app.json`).
- **Rediseño según `frontend/PaltaScan — Pantallas.pdf` (Fase 1, v0.2.0, 2026-09-29).** Implementado todo el diseño salvo la cámara en vivo:
  - arranque con carga del modelo, guía rápida de 3 pasos (solo la primera vez) y barra inferior Inicio / Historial / Lotes / Ajustes;
  - Inicio con saludo, últimos 3 análisis y accesos;
  - Historial agrupado por día con filtros (categoría y madurez) y estado vacío;
  - Resultado con selector original / con marcas, tarjeta OCDE, barra de % de defecto con los límites 9,4 % y 14,1 %, escala de madurez 1–5 con nombres y confianza, aviso de resultado poco confiable y botones Guardar en lote / Compartir / Nuevo análisis; el mismo layout sirve de detalle (Mover a lote / Eliminar con confirmación);
  - Lotes como entidad (nombre, productor, notas y fecha) con barra apilada por categoría, detalle con resumen e histograma de madurez, y exportación a **CSV** y **PDF** (informe A4 con resumen por categoría, histograma de madurez y una fila por análisis con miniatura, generado en el celular con `PdfDocument`);
  - vista previa de galería, pantalla "No pudimos analizar la foto" y Ajustes / Cómo funciona.
- **Diseño básico (v0.3.0, 2026-10-01).** Tras la revisión del docente, que observó que el diseño anterior se veía muy recargado, se pasó a una interfaz sobria de dos pantallas y colores apagados:
  - **Principal:** lote actual (cambiar o crear uno), tomar foto o elegir de la galería, y debajo el resultado del último análisis: imagen con o sin marcas, categoría OCDE, % de defecto, madurez con confianza, tiempo y aviso si la foto es dudosa. Si la foto no sirve, se muestra el motivo.
  - **Historial:** análisis por lote o todos, resumen por categoría y madurez, exportación a CSV y PDF. Al tocar un análisis se abre su detalle, con las opciones de moverlo de lote o eliminarlo.

  La lógica no cambió (inferencia, validación de la captura, base de datos v2, exportadores). El diseño anterior (31 pantallas del PDF) queda guardado en la rama `diseno-material` y en la etiqueta `v0.2.0-diseno-material`.
- **Validación de la captura** (`ml/ValidacionCaptura.kt`), con lo que el modelo ya entrega:
  - Bloquean el análisis, que no se guarda: sin palta, más de una palta (componentes con área ≥ 20% de la mayor) y palta cortada en el borde.
  - Solo generan aviso: foto oscura (luminancia < 70), borrosa (varianza del Laplaciano < 40 a 256 px) o madurez con confianza < 0.6.
  - Los umbrales se calibraron con las 108 fotos de test: el mínimo real es 184 de luminancia y 117 de nitidez, así que ninguna foto válida queda marcada.
- **Base de datos v2:** tabla `lotes` y `inspecciones.lote_id` anulable. Cada análisis va al historial y se asigna a un lote si se desea. La migración desde v1 convierte los lotes de texto en filas.
- 20 pruebas unitarias pasan: las 13 anteriores y 7 de validación de la captura. La inferencia no cambió.
- **Probada en el celular (2026-09-29):**
  - La v0.2.0 se instaló encima de la 0.1.0.
  - `ConcordanciaTest` da resultados idénticos a la v0.1.0: 108/108 en madurez y en categoría, con 708 ms de inferencia.
  - Se recorrieron todas las pantallas con 7 fotos de prueba copiadas a la galería: una por categoría, dos paltas, palta cortada, foto oscura y sin palta.
  - Los tres casos bloqueantes muestran su motivo. La foto oscura también se bloquea como "sin palta", porque el modelo no la detecta con tan poca luz.
  - La exportación a CSV y PDF funciona.
  - Se corrigieron los plurales ("1 lote", "1 palta").
  - La foto "Cat. II" (ratio de 9,4 % en Python) salió Cat. I con 9,1 % en la app: es la foto límite que ya difería en la concordancia.
- **Varios modelos (v0.4.0, 2026-10-02)**, para medir en el celular los modelos de la comparación:
  - `Configuracion.MODELOS`: multitarea (propuesto, umbral de defecto 0.05), U-Net ResNet34 (0.10) y dos redes YOLOv8s-seg + ResNet-34 (0.10), cada uno con sus archivos en assets. Solo se ofrecen los que están instalados.
  - `Analizador` generalizado: en la U-Net, las máscaras salen por umbral por píxel (`Postproceso.mascarasSemanticas`: palta > 0.5, defecto > umbral, recortadas al contenido del letterbox); en las dos redes, el YOLO sin madurez pasa por el mismo postproceso que el multitarea y la madurez sale del clasificador (letterbox a 448 px; el tiempo de inferencia suma ambas redes). Desde las máscaras, todo es igual (fruto completo, ROI, ratio, OCDE, validación de la captura).
  - Pantalla principal: fila "Modelo" con "Cambiar" (solo si hay más de un modelo); cada análisis guarda el modelo (base de datos v3, migración desde v2 con `multitarea` por defecto) y el resultado y el CSV lo indican.
  - `ConcordanciaTest` corre todos los modelos instalados (o uno con el argumento `modelo`) y escribe `resultados_app_<modelo>.json`.
  - 23 pruebas unitarias pasan (3 nuevas: máscaras de la U-Net y lectura de salidas sin madurez).
- Falta:
  - cámara en vivo y modo continuo (fase 2);
  - reducir la latencia (delegado GPU / hilos);
  - instalar los modelos alternativos cuando estén sus pesos.

**Etapa 5: integración.** El modelo cuantizado y el motor OCDE corren completos en el celular. Validación funcional hecha: 20 pruebas unitarias y `ConcordanciaTest` frente a Python (108/108 en madurez, 107/108 en OCDE, ~0.9 s por foto).

La antigua etapa de validación del prototipo (estrés, iluminación, SUS) se retiró del plan; su lugar lo ocupa la comparación de modelos (Etapa 3).

## 13. Pendientes transversales

- **Umbrales OCDE:** el área de referencia de 42.4 cm² se contrastó con mediciones de palta Hass (largo × diámetro × π/4 × factor de forma 0.975 medido en el dataset): entre 40.0 y 46.5 cm² en frutos pequeños (162–207 g) y ~63.3 cm² en calibre comercial (241–267 g). **Análisis de sensibilidad** (`scripts/sensibilidad_ocde.py`, `resultados/sensibilidad_ocde/`): recalcula la categoría real y la predicha con cada área a partir de los `predicciones.csv`, sin volver a inferir. Acierto OCDE del multitarea entre 75.9% y 81.5% (siempre dentro de su IC con 42.4 cm²: 66.1–88.4%); la U-Net es la peor en todas las áreas, así que la selección de la Etapa 3 no depende del supuesto. Con áreas mayores crece la clase Rechazado y el recall de Rechazado del multitarea sube de 51.9% a 78.4%. El umbral de defecto no se toca: se calibró por IoU, no por acierto OCDE. Falta la escala real del montaje para medir cada fruto.

- **Anotaciones de SAM2 a auditar:** en test hay fotos con todo el fruto marcado como defecto (la caja amplia confunde a SAM2). Plan: regla automática contra las cajas originales, revisión visual del test y val y confirmación humana (ver conversación del 2026-09-28).
- **Criterio de cuantización:** falta medir la pérdida de mAP50 de las variantes TFLite.

## 14. Historial de actualizaciones

- 2026-09-26: primera version de este archivo. Inventario del estado real:
  Etapa 1 con ronda 1 de reentrenamiento hecha, Etapa 2 con benchmark de
  cuantizacion completo, Etapas 3-5 sin empezar.
- 2026-09-26: se incorpora el contexto de Seminario I.
  Se registra que la multitarea real ya esta resuelta y se agregan los
  pendientes: umbral OCDE sin fuente, criterio Δ mAP@50 < 3% sin medir,
  dispositivo edge y tecnologia del prototipo sin decidir.
- 2026-09-26: primera version de la app Android `app_movil/PaltaScan/`
  (ver `app_movil/README.md`) y de `scripts/referencia_app.py` para
  compararla con Python. Se decide la app nativa con inferencia on-device.
- 2026-09-28: los modelos estaban en el release de GitHub
  `fvm14/SEMINARIO2` `v0.2-etapa2` (ronda1_best.pt, limpio_best.pt, ONNX y 3
  TFLite, imagenes_anotadas.zip, pseudoetiquetas.zip). Se descargaron
  `ronda1_best.pt` (`weights/`) y el TFLite dynamic-range (`modelos_movil/`),
  con el SHA-256 verificado, y se extrajeron las 108 fotos de prueba en
  `data/prueba_test/` (todo en `.gitignore`). Se crea `.venv` con el runtime
  de TFLite (ai-edge-litert).
- 2026-09-28: verificado el TFLite dynamic-range con `referencia_app.py`
  sobre las 108 fotos de prueba: 73.15% de exactitud de madurez, identica a
  la de `resultados/etapa2` (resultado en
  `resultados/app/referencia_python_test.json`). El modelo ya esta en los
  assets de la app y hay una prueba instrumentada (`ConcordanciaTest`)
  pendiente de correr en un celular real; se descarta el emulador por
  lento. Nuevo `scripts/probar_modelo_gui.py`: ventana tkinter para probar el
  modelo con fotos (original vs. mascara, OCDE, madurez, ms).
- 2026-09-28: se une el `PROGRESO.md` del equipo (secciones 1 a 10) con la bitácora de Dylann (secciones 11 a 13) en un solo archivo, y se marcan como hechos los pendientes de la app.
- 2026-09-28: línea base de Seminario I: modelo descargado, torch 2.14 (CUDA 12.6) y ultralytics 8.4.160 instalados en `.venv`, dataset YOLO regenerado (852/232/108) y umbral calibrado en val. Se detectó que el modelo de Seminario I vio 77 de 108 fotos de test (37 de 39 frutos), así que la comparación directa no es válida; se propone reentrenarlo en el split por fruto (sección 6).
- 2026-09-28: listo el notebook de Colab `notebooks/linea_base_s1_colab.ipynb` y `scripts/train_linea_base_s1.py` (configuraciones `replica_s1` y `ablacion`) para reentrenar la línea base; `eval_linea_base_s1.py` probado localmente.
- 2026-09-28: `probar_modelo_gui.py` v2: selector de modelos, anotación real, métricas por foto y acumuladas de defectos y madurez; verificado contra las métricas de test reportadas.
- 2026-09-28: app probada en el celular (Xiaomi 11 Lite 5G NE, Android 14): se cambió TFLite 2.16.1 por LiteRT 1.4.2 porque no cargaba el modelo; concordancia con Python de 108/108 en madurez y 107/108 en OCDE; latencia de ~0.9 s por foto en CPU.
- 2026-09-29: rediseño de la app (Fase 1, v0.2.0) según el PDF de pantallas: navegación inferior, resultado nuevo, historial por día con filtros, lotes con productor y exportación CSV, guía rápida, ajustes y validación de la captura (sin palta, varias, cortada, oscura, borrosa y baja confianza); base de datos v2 con migración; 20 pruebas unitarias.
- 2026-09-29: exportación del lote a PDF (fase 1b) con `PdfDocument`, sin librerías nuevas.
- 2026-09-29: v0.2.0 probada en el celular: concordancia idéntica (108/108), recorrido completo de pantallas y casos de error, exportación CSV/PDF verificada; plurales corregidos.
- 2026-10-01: app v0.3.0 con diseño básico de 2 pantallas (principal e historial); el diseño anterior se guardó en la rama `diseno-material`.
- 2026-10-01: estilo de la app cambiado a verde y negro con esquinas rectas (barra superior verde, botones planos); se agregó `app_movil/FUNCIONALIDADES.md` con todas las funciones de la app.
- 2026-10-01: verde de la app oscurecido (#1B5E20).
- 2026-10-02: comparación de modelos (YOLOv8s-seg multitarea, U-Net ResNet34 multitarea, YOLOv8s-seg + ResNet-34) incorporada como Etapa 3; el prototipo pasa a ser la Etapa 4 y la integración la 5; se retira la validación del prototipo. `PROGRESO_COMPARACION.md` se unificó en este documento.
- 2026-10-02: análisis de sensibilidad de la categoría OCDE al área de referencia del fruto (40.0, 42.4, 46.5 y 63.3 cm²) para los tres modelos; los límites actuales se mantienen.
- 2026-10-02: app v0.4.0 con soporte para los tres modelos de la comparación (selector de modelo, base de datos v3) y scripts para exportar, cuantizar y evaluar en TFLite la U-Net, el YOLOv8s-seg sin madurez y la ResNet-34; probados con pesos de prueba, a la espera de los pesos reales.
