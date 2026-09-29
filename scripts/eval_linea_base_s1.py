"""
eval_linea_base_s1.py
Evalua el modelo de Seminario I (YOLOv8s-seg de tarea unica: palta y defecto,
sin cabezal de madurez) con el mismo protocolo que eval_yolo_multitarea.py,
para usarlo como linea base del multitarea sobre la particion por fruto.

1) Validacion estandar Ultralytics: P, R, F1, mAP50 y mAP50-95 de mascaras.
2) Analisis por imagen con el umbral de defecto calibrado en validacion
   (calibrar_umbral_defecto.py): IoU por pixel, ratio, OCDE y latencia.

El modelo no predice madurez. Para que predicciones.csv tenga el mismo formato
que el del multitarea (y bootstrap_ic.py --comparar funcione), madurez_pred se
llena con la madurez del nombre del archivo, como hacia Seminario I; esas
columnas no deben reportarse como resultado del modelo.

Uso (desde la raiz del proyecto):
    python scripts/eval_linea_base_s1.py --weights weights/s1_defectos_best.pt --conf-defecto 0.15
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rasterize_utils import poligonos_a_mascaras  # noqa: E402
from metrics import AcumuladorMetricas  # noqa: E402
from ocde import calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi, fruto_completo, CATEGORIAS_OCDE  # noqa: E402
from yolo_multitarea import madurez_desde_archivo  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--conf-defecto", type=float, required=True)
    p.add_argument("--out-dir", default=os.path.join("resultados", "linea_base_s1"))
    p.add_argument("--device", default="")
    args = p.parse_args()
    out_dir = os.path.abspath(os.path.join(args.out_dir, f"eval_{args.split}"))
    os.makedirs(out_dir, exist_ok=True)
    dev = {"device": args.device} if args.device else {}

    modelo = YOLO(args.weights)
    print("Clases del modelo:", modelo.names)

    print("=== 1) Validacion estandar Ultralytics ===")
    m = modelo.val(data=args.data, split=args.split, imgsz=args.imgsz, batch=args.batch,
                   project=out_dir, name="ultralytics", exist_ok=True, plots=True, **dev)
    stats = {k: round(float(v), 4) for k, v in m.results_dict.items()}
    for t in ("B", "M"):
        pr, rc = stats.get(f"metrics/precision({t})", 0), stats.get(f"metrics/recall({t})", 0)
        stats[f"metrics/F1({t})"] = round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0
    por_clase = {}
    for i, nombre in m.names.items():
        mp, mr, m50, m5095 = m.seg.class_result(i)
        por_clase[nombre] = {"P": round(mp, 4), "R": round(mr, 4), "mAP50": round(m50, 4), "mAP50-95": round(m5095, 4)}

    print("\n=== 2) Analisis por imagen ===")
    raiz = os.path.dirname(os.path.abspath(args.data))
    with open(os.path.join(raiz, f"{args.split}.txt"), encoding="utf-8") as f:
        imagenes = [l.strip() for l in f if l.strip()]

    acc_crudo, acc_roi = AcumuladorMetricas(), AcumuladorMetricas()
    filas, latencias = [], []
    for n, ruta in enumerate(imagenes):
        r = modelo.predict(ruta, imgsz=args.imgsz, conf=min(args.conf, args.conf_defecto),
                           retina_masks=True, verbose=False, **dev)[0]
        latencias.append(sum(r.speed.values()))
        alto, ancho = r.orig_shape
        pr_p = np.zeros((alto, ancho), bool)
        pr_d = np.zeros((alto, ancho), bool)
        if r.masks is not None:
            for mk, c, s in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.cls.cpu().numpy().astype(int),
                                r.boxes.conf.cpu().numpy()):
                if c == 0 and s >= args.conf:
                    pr_p |= mk
                elif c == 1 and s >= args.conf_defecto:
                    pr_d |= mk
        pr_p = fruto_completo(pr_p, pr_d)
        pr_d_roi = filtrar_defecto_por_roi(pr_p, pr_d)

        stem = os.path.splitext(os.path.basename(ruta))[0]
        gt_p, gt_d = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), alto, ancho)
        gt_p, gt_d = gt_p.astype(bool), gt_d.astype(bool)
        m_gt = madurez_desde_archivo(ruta)
        t_gt = torch.from_numpy(np.stack([gt_p, gt_d])[None])
        mad = torch.tensor([m_gt])
        acc_crudo.actualizar_binario(torch.from_numpy(np.stack([pr_p, pr_d])[None]), t_gt, mad, mad)
        acc_roi.actualizar_binario(torch.from_numpy(np.stack([pr_p, pr_d_roi])[None]), t_gt, mad, mad)

        r_gt, r_pr = calcular_ratio(gt_p, gt_d), calcular_ratio(pr_p, pr_d_roi)
        filas.append({
            "archivo": os.path.basename(ruta), "madurez_real": m_gt + 1, "madurez_pred": m_gt + 1,
            "confianza_madurez": "", "ratio_real": round(r_gt, 5), "ratio_pred": round(r_pr, 5),
            "ocde_real": CATEGORIAS_OCDE[clasificar_ocde(r_gt)], "ocde_pred": CATEGORIAS_OCDE[clasificar_ocde(r_pr)],
            "latencia_ms": round(latencias[-1], 2),
        })
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(imagenes)}")

    def seg(mm):
        return {k: round(v, 4) for k, v in mm.items() if "madurez" not in k and k != "score"}

    rech = [f for f in filas if f["ocde_real"] == "Rechazado"]
    lat = np.array(latencias[3:])
    resumen = {
        "weights": args.weights, "split": args.split, "n_imagenes": len(filas), "imgsz": args.imgsz,
        "conf_palta": args.conf, "conf_defecto": args.conf_defecto,
        "ultralytics": {"global": stats, "por_clase_mascaras": por_clase},
        "pixel_sin_roi": seg(acc_crudo.calcular()), "pixel_con_roi": seg(acc_roi.calcular()),
        "ocde": {"accuracy": round(float(np.mean([f["ocde_real"] == f["ocde_pred"] for f in filas])), 4),
                 "recall_rechazado": round(float(np.mean([f["ocde_pred"] == "Rechazado" for f in rech])), 4) if rech else None,
                 "mae_ratio": round(float(np.mean([abs(f["ratio_real"] - f["ratio_pred"]) for f in filas])), 5)},
        "latencia_ms": {"media": round(float(lat.mean()), 2), "dispositivo": str(modelo.predictor.device)},
        "nota": "madurez_pred en predicciones.csv es la del nombre del archivo; el modelo no predice madurez.",
    }
    with open(os.path.join(out_dir, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    print(json.dumps({k: resumen[k] for k in ("pixel_con_roi", "ocde")}, indent=2))
    print(f"Mascaras: {stats.get('metrics/precision(M)')} P | {stats.get('metrics/recall(M)')} R | "
          f"mAP50 {stats.get('metrics/mAP50(M)')} | mAP50-95 {stats.get('metrics/mAP50-95(M)')}")
    print(f"Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
