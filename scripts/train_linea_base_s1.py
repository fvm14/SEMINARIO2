"""
train_linea_base_s1.py
Entrena la linea base de tarea unica (YOLOv8s-seg normal: palta y defecto, sin
cabezal de madurez) sobre el split por fruto (data/yolo, 852 imagenes de train).

El best.pt original de Seminario I no sirve como linea base porque se entreno
con el split aleatorio de Roboflow: vio 77 de las 108 fotos de nuestro test
(37 de 39 frutos). Por eso se reentrena aqui con dos configuraciones:

  replica_s1 : hiperparametros exactos de Seminario I (mosaic 1.0,
               hsv 0.015/0.7/0.4, overlap_mask por defecto). Responde "que tan
               bueno era realmente el modelo de Seminario I".
  ablacion   : los mismos hiperparametros que yolov8s_mt_final (sin mosaic,
               hsv suave, overlap_mask=False). La unica diferencia con el
               multitarea es el cabezal de madurez: mide si ese cabezal le
               quita calidad a la segmentacion de defectos.

Uso (desde la raiz del proyecto):
    python scripts/train_linea_base_s1.py --config replica_s1
    python scripts/train_linea_base_s1.py --config ablacion
    python scripts/train_linea_base_s1.py --config ablacion --resume <.../weights/last.pt>
"""

import argparse
import os

from ultralytics import YOLO

CONFIGS = {
    "replica_s1": dict(mosaic=1.0, mixup=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
                       degrees=15.0, flipud=0.5, fliplr=0.5),
    "ablacion": dict(overlap_mask=False, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0, close_mosaic=0,
                     hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
                     degrees=15.0, translate=0.1, scale=0.5, flipud=0.5, fliplr=0.5),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    p.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    p.add_argument("--model", default="yolov8s-seg.pt", help="pesos preentrenados COCO")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="")
    p.add_argument("--project", default=os.path.join("resultados", "linea_base_s1"))
    p.add_argument("--resume", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return
    extra = {"device": args.device} if args.device else {}
    YOLO(args.model).train(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, patience=args.patience,
        workers=args.workers, project=os.path.abspath(args.project), name=f"yolov8s_{args.config}",
        seed=args.seed, deterministic=True, plots=True, exist_ok=False, **CONFIGS[args.config], **extra)


if __name__ == "__main__":
    main()
