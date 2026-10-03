# Sensibilidad de la categoría OCDE al área de referencia

Test: 108 imágenes, 39 frutos. Los límites son 4 cm² / área (Cat. I) y 6 cm² / área (Cat. II); la categoría real y la predicha se recalculan con cada área. IC 95% por bootstrap agrupado por fruto.

## Límites y distribución real

| Área | Límite Cat. I | Límite Cat. II | Cat. I | Cat. II | Rechazado |
|---|---|---|---|---|---|
| Jewe, Etiopia (40.0 cm2) | 10.0% | 15.0% | 72 | 9 | 27 |
| Valor actual (42.4 cm2) | 9.4% | 14.2% | 71 | 9 | 28 |
| Upper Gana, Etiopia (46.5 cm2) | 8.6% | 12.9% | 63 | 16 | 29 |
| Mexico, calibre comercial (63.3 cm2) | 6.3% | 9.5% | 53 | 18 | 37 |

## Acierto OCDE por modelo

| Modelo | Jewe, Etiopia (40.0 cm2) | Valor actual (42.4 cm2) | Upper Gana, Etiopia (46.5 cm2) | Mexico, calibre comercial (63.3 cm2) |
|---|---|---|---|---|
| YOLOv8s-seg multitarea (propuesto) | 76.9% [64.8%, 87.2%] | 77.8% [66.1%, 88.4%] | 75.9% [65.9%, 85.6%] | 81.5% [71.7%, 89.8%] |
| U-Net ResNet34 multitarea | 63.0% [52.1%, 73.8%] | 62.0% [50.5%, 74.1%] | 63.0% [53.1%, 72.7%] | 67.6% [57.5%, 76.6%] |
| YOLOv8s-seg + ResNet-34 | 81.5% [71.3%, 91.0%] | 82.4% [73.3%, 90.6%] | 77.8% [67.3%, 87.5%] | 77.8% [68.2%, 86.4%] |

## F1 macro OCDE

| Modelo | Jewe, Etiopia (40.0 cm2) | Valor actual (42.4 cm2) | Upper Gana, Etiopia (46.5 cm2) | Mexico, calibre comercial (63.3 cm2) |
|---|---|---|---|---|
| YOLOv8s-seg multitarea (propuesto) | 0.595 | 0.654 | 0.678 | 0.778 |
| U-Net ResNet34 multitarea | 0.461 | 0.463 | 0.516 | 0.597 |
| YOLOv8s-seg + ResNet-34 | 0.708 | 0.749 | 0.703 | 0.715 |

## Recall de Rechazado

| Modelo | Jewe, Etiopia (40.0 cm2) | Valor actual (42.4 cm2) | Upper Gana, Etiopia (46.5 cm2) | Mexico, calibre comercial (63.3 cm2) |
|---|---|---|---|---|
| YOLOv8s-seg multitarea (propuesto) | 51.9% | 60.7% | 65.5% | 78.4% |
| U-Net ResNet34 multitarea | 59.3% | 60.7% | 69.0% | 81.1% |
| YOLOv8s-seg + ResNet-34 | 48.1% | 53.6% | 55.2% | 67.6% |
