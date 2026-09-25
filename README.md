# palta-multitarea

Reconstruccion en Python (sin Roboflow) del pipeline de Seminario I, con un
modelo YOLOv8s-seg multitarea que en una sola pasada:

1. Segmenta por instancia la palta y sus defectos (igual que en Seminario I).
2. Clasifica el nivel de madurez 1 a 5 con un cabezal nuevo conectado al
   backbone compartido (capa SPPF).
3. A partir de las mascaras predichas calcula el ratio defecto/palta y la
   categoria OCDE (Categoria I, Categoria II, Rechazado).

## Flujo

| Paso | Script | Reemplaza en Seminario I |
|---|---|---|
| 1. Split train/val/test por fruto | `scripts/split_dataset.py` | split de Roboflow |
| 2. Auditoria de anotaciones (excluye 24 rotas) | `scripts/auditar_anotaciones.py` | (nuevo) |
| 3. Dataset en formato YOLO | `scripts/prepare_yolo.py` | export de Roboflow |
| 4. Entrenamiento multitarea | `scripts/train_yolo_multitarea.py` | `train_yolov8s.py` (Colab) |
| 5. Calibracion del umbral de defecto (val) | `scripts/calibrar_umbral_defecto.py` | umbral fijo 0.35 |
| 6. Evaluacion en test (mAP, madurez, OCDE, ROI, latencia) | `scripts/eval_yolo_multitarea.py` | `evaluar_con_roi.py` + `motor_ocde.py` |
| 7. Intervalos de confianza (bootstrap por fruto) | `scripts/bootstrap_ic.py` | (nuevo) |

## Modelo final

`resultados/yolo_multitarea/yolov8s_mt_peso2/weights/best.pt`, entrenado con:

```powershell
python scripts/train_yolo_multitarea.py --sin-overlap --peso-madurez 2 --name yolov8s_mt_peso2
```

Elegido en validacion frente a la variante con peso de madurez 1 (`yolov8s_mt_limpio`).
Umbrales de inferencia: palta 0.35, defecto 0.15 (calibrado en validacion, max IoU de defecto).

Resultados en test (111 imagenes, 39 frutos; IC 95% por bootstrap por fruto):

| Metrica | Valor | IC 95% |
|---|---|---|
| Mascaras P / R / F1 | 0.783 / 0.761 / 0.772 | |
| Mascaras mAP50 / mAP50-95 | 0.774 / 0.480 | |
| Madurez accuracy | 61.3% | 52.0 - 70.9% |
| Madurez F1 macro | 0.607 | 0.505 - 0.696 |
| Madurez con tolerancia +-1 nivel | 98.2% | 95.2 - 100% |
| Acierto OCDE | 73.0% | 62.8 - 82.1% |
| MAE ratio defecto/palta | 0.072 | 0.050 - 0.097 |
| Latencia (RTX A5000) | 19.7 ms/imagen | |

Frente a `yolov8s_mt_limpio` ninguna diferencia en test es estadisticamente distinguible (bootstrap pareado).

Modulos de apoyo: `yolo_multitarea.py` (modelo, trainer y validador),
`ocde.py` (umbrales OCDE y filtro ROI de 20 px), `rasterize_utils.py`
(poligonos a mascaras), `metrics.py` (IoU/Dice por pixel).

## Instalacion (PowerShell, una sola vez)

```powershell
cd C:\Users\pc\Documents\palta-multitarea
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

La ultima linea debe imprimir `True NVIDIA RTX A5000 Laptop GPU`.

## Uso (desde la raiz del proyecto, con el entorno activado)

```powershell
python scripts/prepare_yolo.py                                          # una sola vez
python scripts/train_yolo_multitarea.py --epochs 2 --fraction 0.1      # prueba rapida
python scripts/train_yolo_multitarea.py                                 # entrenamiento completo
python scripts/eval_yolo_multitarea.py --weights resultados\yolo_multitarea\yolov8s_multitarea\weights\best.pt
```

Si se interrumpe: `python scripts/train_yolo_multitarea.py --resume resultados\yolo_multitarea\yolov8s_multitarea\weights\last.pt`.
Si la prueba rapida creo la carpeta `yolov8s_multitarea`, el entrenamiento
completo se guardara como `yolov8s_multitarea2` (Ultralytics no sobrescribe).

## Decisiones de diseno

- **Split por fruto.** El dataset fotografia la misma palta varios dias y por
  ambos lados (415 frutos, 1,243 imagenes). Se agrupa por fruto
  (StratifiedGroupKFold, semilla 42) para que ninguna palta aparezca en dos
  subconjuntos, y se balancea por madurez. Train 892 imagenes / 297 frutos,
  val 239 / 79, test 112 / 39. El split de Roboflow de Seminario I fue por
  imagen, por lo que sus metricas probablemente estan infladas por esta fuga.
- **Hiperparametros de Seminario I** (imgsz 800, 100 epocas, batch 8,
  patience 20, pesos COCO de yolov8s-seg.pt) salvo:
  - sin mosaic/mixup/cutmix/copy-paste: mezclarian paltas de distinta madurez
    en una imagen con una sola etiqueta;
  - hsv_s 0.4 y hsv_v 0.3 (antes 0.7 y 0.4): la madurez Hass se ve en el color
    de la cascara y una variacion de color agresiva borra esa senal.
- **Perdida:** perdida YOLO (box, seg, cls, dfl) + `peso_madurez` x
  CrossEntropy(madurez), `--peso-madurez 1.0` por defecto.
- **Seleccion de best.pt:** `0.5 x fitness de mascaras + 0.5 x F1 macro de madurez`.
- **Etiqueta de madurez:** se lee del nombre del archivo
  (`T10_d01_033_a_<madurez>_...`), igual que `motor_ocde.extraer_madurez`.
- Se excluye 1 imagen sin poligono de palta (`T10_d04_133_a_1`) y 24 con anotaciones
  rotas por fallos de SAM2 (palta diminuta o rectangulo de respaldo como defecto), listadas
  en `data/splits/anotaciones_excluidas.csv`. El split no cambia: se excluyen sin re-sortear.
- `overlap_mask=False` (`--sin-overlap`): con el default de Ultralytics la mascara de palta
  queda con huecos donde hay defectos, mientras el poligono anotado cubre el fruto completo.
- Umbral por clase: el defecto usa un umbral mas bajo que la palta, elegido en validacion.

## Pendiente para Seminario II

La exportacion a movil (ONNX/TFLite) necesita un envoltorio que devuelva
tambien los logits de madurez, porque el exportador estandar de Ultralytics
solo exporta la salida de deteccion/segmentacion.
