"""
calibrar_regla_pseudo.py
Elige la regla de aceptacion automatica de pseudo-etiquetas usando el split de
VALIDACION (tiene anotacion humana, asi que se puede medir la calidad de lo
que se aceptaria), en vez de revisar miles de fotos a mano.

Regla evaluada (por foto):
  - defectos incluidos: instancias con confianza >= conf_def
  - "area dudosa": fraccion del area de defecto que viene de instancias con
    confianza < 0.5
  - la foto se ACEPTA si area dudosa <= max_dudosa y no hay defectos rectangulares
Para cada combinacion (conf_def, max_dudosa) se reporta sobre las fotos aceptadas:
  % de fotos aceptadas, precision / recall / IoU de defecto por pixel,
  acierto OCDE y fotos con defecto real donde la pseudo-etiqueta no marca nada.

Uso:
    python scripts/calibrar_regla_pseudo.py --weights resultados/yolo_multitarea/yolov8s_mt_final/weights/best.pt
Por partes: --inicio 0 --fin 120 ; luego --solo-resumen
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_multitarea  # noqa: E402,F401
from rasterize_utils import poligonos_a_mascaras  # noqa: E402
from ocde import calcular_ratio, clasificar_ocde  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CONF_DEF = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
MAX_DUDOSA = [0.0, 0.10, 0.25, 0.50, 1.0]
ALTA = 0.5


def es_rectangular(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return False
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return w * h > 400 and cv2.contourArea(c) / (w * h) > 0.9


def resumir(ruta):
    d = pd.read_csv(ruta).drop_duplicates(subset=["archivo", "conf_def", "max_dudosa"], keep="last")
    filas = []
    for (cd, md), g in d.groupby(["conf_def", "max_dudosa"]):
        a = g[g.aceptada == 1]
        tp, fp, fn = a.tp.sum(), a.fp.sum(), a.fn.sum()
        filas.append({"conf_def": cd, "max_dudosa": md, "aceptadas_%": 100 * len(a) / len(g),
                      "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                      "iou": tp / max(1, tp + fp + fn), "acierto_ocde": (a.ocde_real == a.ocde_pred).mean(),
                      "defecto_omitido": int(((a.def_real > 0) & (a.def_pred == 0)).sum()), "n": len(g)})
    r = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    r.to_csv(ruta.replace(".csv", "_resumen.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    ap.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--inicio", type=int, default=0)
    ap.add_argument("--fin", type=int, default=0)
    ap.add_argument("--salida", default="")
    ap.add_argument("--solo-resumen", action="store_true")
    args = ap.parse_args()
    salida = args.salida or os.path.join(os.path.dirname(os.path.dirname(args.weights)), "calibracion_regla_pseudo_val.csv")
    if args.solo_resumen:
        return resumir(salida)

    with open(os.path.join(os.path.dirname(os.path.abspath(args.data)), "val.txt"), encoding="utf-8") as f:
        imgs = [l.strip() for l in f if l.strip()][args.inicio: args.fin or None]
    modelo = YOLO(args.weights)
    nuevo = not os.path.exists(salida)
    with open(salida, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["archivo", "conf_def", "max_dudosa", "aceptada", "tp", "fp", "fn",
                                           "def_real", "def_pred", "ocde_real", "ocde_pred"])
        if nuevo:
            w.writeheader()
        for ruta in imgs:
            r = modelo.predict(ruta, imgsz=args.imgsz, conf=min(CONF_DEF), retina_masks=True, verbose=False)[0]
            H, W = r.orig_shape
            stem = os.path.splitext(os.path.basename(ruta))[0]
            gp, gd = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), H, W)
            gp, gd = gp.astype(bool), gd.astype(bool)
            c_gt = clasificar_ocde(calcular_ratio(gp, gd))
            palta = np.zeros((H, W), bool)
            defs = []
            if r.masks is not None:
                ms = r.masks.data.cpu().numpy() > 0.5
                cl = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                ip = [i for i in range(len(cl)) if cl[i] == 0 and cf[i] >= 0.35]
                if ip:
                    palta = ms[max(ip, key=lambda j: cf[j])]
                for i in range(len(cl)):
                    if cl[i] == 1 and (not palta.any() or (ms[i] & palta).sum() >= 0.5 * ms[i].sum()):
                        defs.append((cf[i], ms[i], es_rectangular(ms[i])))
            for cd in CONF_DEF:
                sel = [x for x in defs if x[0] >= cd]
                pred = np.zeros((H, W), bool)
                dud = np.zeros((H, W), bool)
                for c, m, _ in sel:
                    pred |= m
                    if c < ALTA:
                        dud |= m
                frac_dud = (dud & pred).sum() / max(1, pred.sum())
                rect = any(x[2] for x in sel)
                c_pr = clasificar_ocde(calcular_ratio(palta | pred, pred))
                for md in MAX_DUDOSA:
                    w.writerow({"archivo": os.path.basename(ruta), "conf_def": cd, "max_dudosa": md,
                                "aceptada": int(frac_dud <= md + 1e-9 and not rect),
                                "tp": int((pred & gd).sum()), "fp": int((pred & ~gd).sum()), "fn": int((~pred & gd).sum()),
                                "def_real": int(gd.sum()), "def_pred": int(pred.sum()),
                                "ocde_real": c_gt, "ocde_pred": c_pr})
    if not args.fin:
        resumir(salida)


if __name__ == "__main__":
    main()
