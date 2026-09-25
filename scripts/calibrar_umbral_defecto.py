"""
calibrar_umbral_defecto.py
Elige el umbral de confianza para la clase defecto usando el split de
VALIDACION (nunca test, para no inflar los resultados reportados).

Cada imagen se predice una sola vez con un umbral bajo; luego, para cada
umbral candidato, se conservan solo los defectos con confianza >= umbral y se
calcula: IoU/precision/recall de defecto por pixel, acierto OCDE y recall de
la categoria Rechazado (el error mas costoso: dejar pasar fruta danada).
La palta mantiene su umbral operativo (0.35).

Uso (desde la raiz del proyecto):
    python scripts/calibrar_umbral_defecto.py --weights resultados/yolo_multitarea/<run>/weights/best.pt
Por partes (si hace falta):
    python scripts/calibrar_umbral_defecto.py --weights ... --inicio 0 --fin 80
    python scripts/calibrar_umbral_defecto.py --weights ... --solo-resumen
"""

import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_multitarea  # noqa: E402,F401  (necesario para cargar el checkpoint)
from rasterize_utils import poligonos_a_mascaras  # noqa: E402
from ocde import calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi, fruto_completo, CATEGORIAS_OCDE  # noqa: E402
from ultralytics import YOLO  # noqa: E402

UMBRALES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def resumir(ruta_csv):
    d = pd.read_csv(ruta_csv).drop_duplicates(subset=["archivo", "umbral"], keep="last")
    filas = []
    for u, g in d.groupby("umbral"):
        tp, fp, fn = g.tp.sum(), g.fp.sum(), g.fn.sum()
        rech = g[g.ocde_real == "Rechazado"]
        filas.append({
            "umbral": u, "n": len(g),
            "iou_defecto": tp / max(1, tp + fp + fn),
            "precision_def": tp / max(1, tp + fp), "recall_def": tp / max(1, tp + fn),
            "acc_ocde": (g.ocde_real == g.ocde_pred).mean(),
            "recall_rechazado": (rech.ocde_pred == "Rechazado").mean() if len(rech) else float("nan"),
            "rechazado_como_catI": (rech.ocde_pred == "Categoria I").sum(),
            "mae_ratio": (g.ratio_real - g.ratio_pred).abs().mean(),
        })
    r = pd.DataFrame(filas)
    print(r.round(4).to_string(index=False))
    # Criterio: max IoU de defecto por pixel. No se usa el acierto OCDE directo porque
    # ~64% de las paltas son Categoria I y ese desbalance premia umbrales altos que
    # dejan pasar fruta rechazable como Categoria I.
    mejor = r.sort_values(["iou_defecto", "acc_ocde"], ascending=False).iloc[0]
    print(f"\nUmbral recomendado para defecto (max IoU de defecto, {int(mejor.n)} imagenes de validacion): {mejor.umbral}")
    r.to_csv(ruta_csv.replace(".csv", "_resumen.csv"), index=False)
    return mejor.umbral


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    p.add_argument("--split", default="val")
    p.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--conf-palta", type=float, default=0.35)
    p.add_argument("--inicio", type=int, default=0)
    p.add_argument("--fin", type=int, default=0)
    p.add_argument("--salida", default="")
    p.add_argument("--solo-resumen", action="store_true")
    p.add_argument("--device", default="")
    args = p.parse_args()

    salida = args.salida or os.path.join(os.path.dirname(os.path.dirname(args.weights)),
                                         f"calibracion_defecto_{args.split}.csv")
    if args.solo_resumen:
        resumir(salida)
        return

    raiz = os.path.dirname(os.path.abspath(args.data))
    with open(os.path.join(raiz, f"{args.split}.txt"), encoding="utf-8") as f:
        imagenes = [l.strip() for l in f if l.strip()]
    imagenes = imagenes[args.inicio: args.fin or None]

    modelo = YOLO(args.weights)
    nuevo = not os.path.exists(salida)
    with open(salida, "a", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=["archivo", "umbral", "tp", "fp", "fn", "ratio_real",
                                             "ratio_pred", "ocde_real", "ocde_pred"])
        if nuevo:
            w.writeheader()
        for n, ruta in enumerate(imagenes):
            r = modelo.predict(ruta, imgsz=args.imgsz, conf=min(UMBRALES), retina_masks=True, verbose=False,
                               **({"device": args.device} if args.device else {}))[0]
            alto, ancho = r.orig_shape
            stem = os.path.splitext(os.path.basename(ruta))[0]
            gt_p, gt_d = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), alto, ancho)
            gt_p, gt_d = gt_p.astype(bool), gt_d.astype(bool)
            r_gt = calcular_ratio(gt_p, gt_d)

            if r.masks is not None:
                mascaras = r.masks.data.cpu().numpy() > 0.5
                clases = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
            else:
                mascaras, clases, confs = np.zeros((0, alto, ancho), bool), np.zeros(0, int), np.zeros(0)

            palta = np.zeros((alto, ancho), bool)
            for mk, c, s in zip(mascaras, clases, confs):
                if c == 0 and s >= args.conf_palta:
                    palta |= mk
            for u in UMBRALES:
                defecto = np.zeros((alto, ancho), bool)
                for mk, c, s in zip(mascaras, clases, confs):
                    if c == 1 and s >= u:
                        defecto |= mk
                fruto = fruto_completo(palta, defecto)
                defecto = filtrar_defecto_por_roi(fruto, defecto)
                r_pr = calcular_ratio(fruto, defecto)
                w.writerow({"archivo": os.path.basename(ruta), "umbral": u,
                            "tp": int((defecto & gt_d).sum()), "fp": int((defecto & ~gt_d).sum()),
                            "fn": int((~defecto & gt_d).sum()),
                            "ratio_real": round(r_gt, 5), "ratio_pred": round(r_pr, 5),
                            "ocde_real": CATEGORIAS_OCDE[clasificar_ocde(r_gt)],
                            "ocde_pred": CATEGORIAS_OCDE[clasificar_ocde(r_pr)]})
            if (n + 1) % 20 == 0:
                print(f"  {args.inicio + n + 1} procesadas")
    if not args.fin:
        resumir(salida)


if __name__ == "__main__":
    main()
