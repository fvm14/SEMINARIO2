# Comparación de modelos (test: 108 imágenes, 39 frutos)

Mismos datos para los tres: 5,475 imágenes de entrenamiento (852 anotadas + 4,623 pseudoetiquetas), validación de 232 y prueba de 108 con anotación humana, partición por fruto, semilla 42. Umbral de defecto calibrado en validación para cada modelo. IC 95% por bootstrap agrupado por fruto (5,000 remuestreos).

| Métrica | YOLOv8s-seg multitarea (propuesto) | U-Net ResNet34 multitarea | Dos redes: YOLOv8s-seg + ResNet-34 |
|---|---|---|---|
| IoU de defecto (píxel) | 0.397 | 0.388 | 0.443 |
| Acierto OCDE | 77.8% [65.7, 87.9] | 62.0% [50.0, 73.9] | 82.4% [73.0, 90.9] |
| F1 macro OCDE | 0.654 | 0.463 | 0.749 |
| Recall de Rechazado | 60.7% | 60.7% | 53.6% |
| MAE del ratio | 0.075 | 0.074 | 0.065 |
| Exactitud de madurez | 74.1% [67.3, 81.0] | 73.1% [64.2, 81.7] | 77.8% [69.7, 85.3] |
| F1 macro de madurez | 0.737 | 0.735 | 0.772 |
| Parámetros | 11.9 M | 24.6 M | 33.1 M (11.8 + 21.3) |
| Tamaño FP32 | 48 MB | 98 MB | 132 MB |
| Latencia de inferencia en CPU | 416 ms | 856 ms | 584 ms (422 + 162) |
| Umbral de defecto | 0.05 | 0.03 | 0.10 |

U-Net con el umbral recalibrado en validación (0.03; antes 0.10, que quedaba en el borde del barrido). Latencia: solo inferencia, PyTorch en CPU, misma máquina, entrada de 800 px (448 px el clasificador), media de 15 pasadas.

## Diferencias frente al modelo propuesto (bootstrap pareado)

| Métrica | U-Net menos propuesto | Dos redes menos propuesto |
|---|---|---|
| Acierto OCDE | -15.7 puntos [-27.6, -3.5] significativa | +4.6 puntos [-1.7, +11.1] no significativa |
| F1 macro OCDE | -0.191 [-0.317, -0.036] significativa | +0.095 [+0.008, +0.198] significativa |
| Recall de Rechazado | 0.0 puntos [-20.8, +23.8] no significativa | -7.1 puntos [-22.9, +7.1] no significativa |
| Exactitud de madurez | -0.9 puntos [-8.8, +6.3] no significativa | +3.7 puntos [-4.0, +11.7] no significativa |
| F1 macro de madurez | -0.002 [-0.081, +0.071] no significativa | +0.035 [-0.049, +0.117] no significativa |
| MAE del ratio | -0.001 [-0.027, +0.021] no significativa | -0.010 [-0.028, +0.005] no significativa |
