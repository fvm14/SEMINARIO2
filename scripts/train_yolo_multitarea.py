"""
train_yolo_multitarea.py
Entrena YOLOv8s-seg + cabezal de madurez (ver yolo_multitarea.py), en local,
sin Roboflow.

Hiperparametros alineados con train_yolov8s.py de Seminario I (imgsz 800,
100 epocas, batch 8, patience 20) con estos cambios deliberados:
  - mosaic=0, mixup=0, cutmix=0, copy_paste=0: mezclan varias imagenes en
    una, lo que dejaria una imagen con pixeles de paltas de distinta madurez
    y una sola etiqueta de madurez.
  - hsv_s y hsv_v mas suaves (0.4 / 0.3 en vez de 0.7 / 0.4): la madurez de
    la palta Hass se ve sobre todo en el color de la cascara (verde a
    morado/negro); una variacion de color muy agresiva borra esa senal.

Uso (desde la raiz del proyecto):
    python scripts/prepare_yolo.py                       # una sola vez
    python scripts/train_yolo_multitarea.py              # entrenamiento completo
    python scripts/train_yolo_multitarea.py --epochs 2 --fraction 0.1   # prueba rapida
    python scripts/train_yolo_multitarea.py --resume resultados/yolo_multitarea/<run>/weights/last.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_multitarea import TrainerMultitarea  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    p.add_argument("--model", default="yolov8s-seg.pt", help="pesos preentrenados COCO")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--peso-madurez", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="")
    p.add_argument("--fraction", type=float, default=1.0, help="fraccion del train (pruebas)")
    p.add_argument("--project", default=os.path.join("resultados", "yolo_multitarea"))
    p.add_argument("--name", default="yolov8s_multitarea")
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sin-overlap", action="store_true",
                   help="overlap_mask=False: cada instancia tiene su propia mascara, asi la palta "
                        "aprende el fruto completo (como el poligono anotado) en vez de quedar con "
                        "huecos donde hay defectos")
    args = p.parse_args()

    TrainerMultitarea.peso_madurez = args.peso_madurez

    if args.resume:
        overrides = {"model": args.resume, "resume": args.resume}
    else:
        overrides = dict(
            model=args.model, data=args.data, imgsz=args.imgsz, epochs=args.epochs,
            batch=args.batch, patience=args.patience, workers=args.workers,
            fraction=args.fraction, project=os.path.abspath(args.project), name=args.name,
            seed=args.seed, deterministic=True, plots=True, exist_ok=False,
            overlap_mask=not args.sin_overlap,
            # augmentations
            mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
            hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
            degrees=15.0, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5,
        )
        if args.device:
            overrides["device"] = args.device

    trainer = TrainerMultitarea(overrides=overrides)
    trainer.train()
    print(f"\nListo. Pesos en: {trainer.save_dir / 'weights'}")


if __name__ == "__main__":
    main()
