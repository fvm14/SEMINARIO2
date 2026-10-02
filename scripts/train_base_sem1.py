"""
train_base_sem1.py
Linea base para la comparacion: el modelo ORIGINAL de Seminario I
(YOLOv8s-seg estandar, sin cabezal de madurez) entrenado con la misma
configuracion de Modelos/YOLOv8s/train_yolov8s.py, pero sobre el mismo volumen
de datos que el modelo multitarea de la Ronda 1 (data/yolo_ampliado: 5,475
imagenes de train = 852 anotadas + 4,623 pseudoetiquetas; val y test iguales).

Se conserva TODO lo de Seminario I: pesos COCO yolov8s-seg.pt, imgsz 800,
100 epocas, batch 8, patience 20, mosaic 1.0, hsv 0.015/0.7/0.4, volteos 0.5,
rotacion 15, traslacion 0.1, escala 0.5 y overlap_mask por defecto.
Lo unico que cambia frente a Seminario I son los datos y el split (por fruto).

Uso (desde la raiz del proyecto, con el entorno activado):
    python scripts/train_base_sem1.py
    python scripts/train_base_sem1.py --epochs 2 --fraction 0.05 --name prueba   # prueba rapida
    python scripts/train_base_sem1.py --resume resultados/yolo_multitarea/yolov8s_base_sem1/weights/last.pt
"""

import argparse
import os

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "yolo_ampliado", "data.yaml"))
    p.add_argument("--model", default="yolov8s-seg.pt")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--fraction", type=float, default=1.0)
    p.add_argument("--project", default=os.path.join("resultados", "yolo_multitarea"))
    p.add_argument("--name", default="yolov8s_base_sem1")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return

    modelo = YOLO(args.model)
    modelo.train(
        data=args.data, task="segment", epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
        patience=args.patience, workers=args.workers, fraction=args.fraction,
        project=os.path.abspath(args.project), name=args.name, exist_ok=False,
        seed=args.seed, deterministic=True, save=True, save_period=10, plots=True,
        # aumentacion de Seminario I (train_yolov8s.py)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.5, fliplr=0.5,
        degrees=15.0, translate=0.1, scale=0.5, mosaic=1.0, mixup=0.0,
    )
    print(f"\nListo. Pesos en: {os.path.join(os.path.abspath(args.project), args.name, 'weights')}")


if __name__ == "__main__":
    main()
