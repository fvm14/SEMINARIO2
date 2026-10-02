# Progreso: comparación de modelos de visión computacional

Resumen del trabajo realizado entre el 1 y el 2 de octubre de 2026, desde que el asesor pidió reemplazar la etapa de validación por una comparación con otros modelos de visión computacional. Complementa a `PROGRESO.md`, que cubre las Etapas 1 y 2.

## 1. Motivo y alcance

El asesor solicitó que, en lugar de la validación del prototipo, se compare el modelo utilizado con otros modelos de visión computacional. El propósito de la comparación es sustentar de forma objetiva la elección del modelo. Por ello se comparan métricas y no funcionamiento: los modelos alternativos se entrenan y se evalúan, pero no se cuantizan ni se integran en la aplicación.

## 2. Decisiones tomadas

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

## 3. Modelos entrenados

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

## 4. Resultados en el conjunto de prueba

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

## 5. Diferencias frente al modelo propuesto (bootstrap pareado)

| Métrica | U-Net menos propuesto | Dos redes menos propuesto |
|---|---|---|
| Acierto OCDE | -12.0 puntos [-20.3, -3.4], significativa | +4.6 puntos [-1.7, +11.1], no significativa |
| F1 macro OCDE | -0.187 [-0.287, -0.058], significativa | +0.095 [+0.008, +0.198], significativa |
| Recall de Rechazado | -14.3 puntos [-27.6, 0.0], no significativa | -7.1 puntos [-22.9, +7.1], no significativa |
| Exactitud de madurez | -0.9 puntos [-8.8, +6.3], no significativa | +3.7 puntos [-4.0, +11.7], no significativa |
| F1 macro de madurez | -0.002 [-0.081, +0.071], no significativa | +0.035 [-0.049, +0.117], no significativa |
| MAE del ratio | -0.009 [-0.034, +0.011], no significativa | -0.010 [-0.028, +0.005], no significativa |

## 6. Interpretación

**U-Net frente al propuesto.** Ambos empatan en IoU de defecto y en madurez, pero la U-Net acierta la categoría OCDE 12 puntos menos, con diferencia significativa, y requiere el doble de parámetros y de latencia. Queda descartada.

**Dos redes frente al propuesto.** Las dos redes obtienen los mejores valores en casi todas las métricas de desempeño. Las diferencias en acierto OCDE y en madurez no son significativas; la del F1 macro de OCDE sí lo es, a favor de las dos redes. El propuesto conserva mejor recall de Rechazado, sin significancia. A cambio, las dos redes ocupan 2.8 veces más y tardan 40% más.

**Conclusión.** La comparación sustenta la elección por equilibrio, no por superioridad. El modelo multitarea no es significativamente inferior al mejor en acierto OCDE ni en madurez, y es el de menor tamaño y latencia, por lo que cumple el criterio de selección. No corresponde afirmar que es el más exacto: cede una ventaja pequeña en exactitud a cambio de una ganancia grande en eficiencia, que es lo que pesa en un dispositivo móvil.

**Efecto de agregar la madurez.** La comparación entre el YOLOv8s-seg sin madurez y el multitarea indica que la segunda tarea tiene un costo pequeño en segmentación (IoU de defecto de 0.443 a 0.397; acierto OCDE de 82.4% a 77.8%, no significativo). Las dos configuraciones difieren además en los aumentos (mosaic y color), por lo que el efecto no puede atribuirse solo al cabezal.

## 7. Limitaciones

- Una sola corrida por modelo, con semilla 42. No se midió la variación entre semillas.
- El conjunto de prueba es pequeño (108 imágenes, 39 frutos), lo que da intervalos amplios.
- El umbral de la U-Net (0.10) coincidió con el mínimo del barrido original. Se agregó 0.05 al barrido; falta repetir la calibración para confirmar.
- El clasificador ResNet-34 usa entrada de 448 px, frente a 800 px de los modelos de segmentación.
- La latencia se midió en computadora y sin cuantizar. Solo el modelo propuesto tiene medición en teléfono.
- Todas las imágenes provienen de condiciones controladas de iluminación.

## 8. Cambios en el documento de tesis

- **Objetivo general:** se retiró "en condiciones reales de operación" y se reemplazó "segmentación de instancias, condición necesaria" por "segmentación, condición necesaria [...] y emplea para ello un modelo de segmentación de instancias", porque la segmentación semántica también permite medir el área.
- **Párrafos de objetivo:** se redactó un párrafo inicial para cada etapa y cada actividad de la segunda fase.
- **Pendiente:** reescribir la Etapa 5 como comparación de modelos, con su objetivo, el criterio de selección y las tablas de las secciones 4 y 5. La actividad 1.3 aún lista como pendiente la línea base sin madurez, que ya está entrenada.

## 9. Archivos

| Qué | Dónde |
|---|---|
| Tabla comparativa | `resultados/comparacion/tabla_comparativa.md` |
| U-Net: pesos, curvas y evaluación | `resultados/comparacion/unet_resnet34/` |
| Clasificador: pesos y curvas | `resultados/comparacion/resnet34_madurez/` |
| Dos redes: predicciones y métricas combinadas | `resultados/comparacion/dos_redes/` |
| YOLOv8s-seg sin madurez | `resultados/yolo_multitarea/yolov8s_base_sem1/` |
| Modelo propuesto | `resultados/yolo_multitarea/yolov8s_mt_ronda1/` |
| Scripts nuevos | `scripts/unet_multitarea.py`, `scripts/eval_unet_multitarea.py`, `scripts/clasificador_madurez.py`, `scripts/train_base_sem1.py` |

## 10. Pendientes

1. Repetir la calibración de la U-Net con el barrido ampliado y reevaluar si cambia el umbral:
   `python scripts/eval_unet_multitarea.py --weights resultados\comparacion\unet_resnet34\best.pt --calibrar`
2. Redactar la sección de comparación en el documento.
3. Opcional: agregar Mask R-CNN como cuarto modelo.
4. Opcional: reentrenar el multitarea con mosaic activado, para comprobar si recupera la diferencia de segmentación frente al modelo sin madurez.
5. Subir los cambios al repositorio (los pesos `.pt` quedan fuera por el `.gitignore`).
