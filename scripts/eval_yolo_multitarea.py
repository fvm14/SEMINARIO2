"""
eval_yolo_multitarea.py
Evalua un best.pt de YOLOv8s-seg multitarea sobre el split de test (o val).

1) Validacion estandar Ultralytics -> mismas metricas que la Tabla 6.1 de
   Seminario I: Precision, Recall, F1, mAP50, mAP50-95 (cajas y mascaras),
   por clase, mas accuracy / F1 macro / matriz de confusion de madurez.
2) Analisis por imagen con el umbral operativo (conf 0.35, como
   evaluar_con_roi.py):
   - mascaras palta/defecto predichas (con y sin filtro ROI de 20 px)
   - IoU/Dice por pixel contra los poligonos anotados
   - ratio defecto/palta y categoria OCDE predicha vs real
   - madurez predicha y su confianza
   - latencia por imagen (preproceso + inferencia + postproceso)
   - visualizaciones real vs prediccion

Uso (desde la raiz del proyecto):
    python scripts/eval_yolo_multitarea.py --weights resultados/yolo_multitarea/yolov8s_multitarea/weights/best.pt
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_multitarea import ValidadorMultitarea, buscar_modelo_multitarea, madurez_desde_archivo  # noqa: E402
from rasterize_utils import poligonos_a_mascaras  # noqa: E402
from metrics import AcumuladorMetricas  # noqa: E402
from ocde import (calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi,  # noqa: E402
                  mayor_componente, fruto_completo, CATEGORIAS_OCDE, KERNEL_DILATACION_ROI)
from ultralytics import YOLO  # noqa: E402

NOMBRES_MADUREZ = ["1", "2", "3", "4", "5"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.35, help="umbral operativo (CONF_OPERATIVO de Seminario I)")
    p.add_argument("--conf-defecto", type=float, default=0.25,
                   help="umbral para la clase defecto, calibrado en validacion (calibrar_umbral_defecto.py)")
    p.add_argument("--kernel-roi", type=int, default=KERNEL_DILATACION_ROI)
    p.add_argument("--n-vis", type=int, default=20)
    p.add_argument("--max-imagenes", type=int, default=0, help="solo para pruebas")
    p.add_argument("--device", default="")
    p.add_argument("--out-dir", default="")
    p.add_argument("--sin-validacion", action="store_true", help="omite el paso 1 (mAP de Ultralytics)")
    return p.parse_args()


def graficar_matriz(matriz, etiquetas, titulo, ruta):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=30, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    corte = matriz.max() / 2 if matriz.max() else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, int(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > corte else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def overlay(img_bgr, palta, defecto):
    """Relleno suave + contorno, para que se siga viendo la cascara debajo."""
    out = img_bgr.astype(np.float32).copy()
    out[palta] = out[palta] * 0.8 + np.array([0, 200, 0]) * 0.2
    out[defecto] = out[defecto] * 0.55 + np.array([0, 0, 255]) * 0.45
    out = out.astype(np.uint8)
    for m, color, grosor in ((palta, (0, 170, 0), 2), (defecto, (0, 0, 255), 2)):
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, color, grosor)
    return out


def _rotulo(panel, lineas):
    y = 32
    for t in lineas:
        cv2.putText(panel, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(panel, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
        y += 32


def guardar_vis(ruta, img_bgr, gt_p, gt_d, pr_p, pr_d, txt_gt, txt_pr):
    """Tres paneles: original | anotacion real | prediccion del modelo."""
    original = img_bgr.copy()
    _rotulo(original, ["ORIGINAL"])
    real = overlay(img_bgr, gt_p, gt_d)
    _rotulo(real, ["ANOTACION REAL"] + txt_gt.replace("REAL  ", "").split(" | "))
    pred = overlay(img_bgr, pr_p, pr_d)
    _rotulo(pred, ["MODELO"] + txt_pr.replace("PRED  ", "").split(" | "))
    sep = np.full((img_bgr.shape[0], 8, 3), 255, np.uint8)
    lienzo = np.concatenate([original, sep, real, sep, pred], axis=1)
    lienzo = cv2.resize(lienzo, (lienzo.shape[1] * 2 // 3, lienzo.shape[0] * 2 // 3), interpolation=cv2.INTER_AREA)
    cv2.imwrite(ruta, lienzo, [cv2.IMWRITE_JPEG_QUALITY, 88])


def validacion_ultralytics(args, out_dir):
    validador = ValidadorMultitarea(args=dict(
        model=args.weights, data=args.data, split=args.split, imgsz=args.imgsz,
        batch=args.batch, plots=True, project=out_dir, name="ultralytics", exist_ok=True,
        **({"device": args.device} if args.device else {})))
    stats = validador()
    m = validador.metrics
    por_clase = {}
    for i, nombre in m.names.items():
        bp, br, b50, b5095 = m.box.class_result(i) if hasattr(m.box, "class_result") else (None,) * 4
        mp, mr, m50, m5095 = m.seg.class_result(i)
        por_clase[nombre] = {
            "box": {"P": bp, "R": br, "mAP50": b50, "mAP50-95": b5095},
            "mask": {"P": mp, "R": mr, "mAP50": m50, "mAP50-95": m5095},
        }
    for tipo in ("B", "M"):
        p, r = stats.get(f"metrics/precision({tipo})", 0), stats.get(f"metrics/recall({tipo})", 0)
        stats[f"metrics/F1({tipo})"] = 2 * p * r / (p + r) if p + r else 0.0
    return {k: round(float(v), 4) for k, v in stats.items()}, por_clase, validador.matriz_madurez


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(args.weights)),
        f"eval_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    vis_dir = os.path.join(out_dir, "visualizaciones")
    os.makedirs(vis_dir, exist_ok=True)

    print("=== 1) Validacion estandar (mAP, como Seminario I) ===")
    stats, por_clase = {}, {}
    if not args.sin_validacion:
        stats, por_clase, _ = validacion_ultralytics(args, out_dir)

    print("\n=== 2) Analisis por imagen (OCDE, pixel IoU, latencia) ===")
    raiz = os.path.dirname(os.path.abspath(args.data))
    with open(os.path.join(raiz, f"{args.split}.txt"), encoding="utf-8") as f:
        imagenes = [l.strip() for l in f if l.strip()]
    if args.max_imagenes:
        imagenes = imagenes[:args.max_imagenes]

    modelo = YOLO(args.weights)
    acc_crudo, acc_roi = AcumuladorMetricas(), AcumuladorMetricas()
    filas, ocde_gt, ocde_pr, latencias = [], [], [], []

    for n, ruta in enumerate(imagenes):
        r = modelo.predict(ruta, imgsz=args.imgsz, conf=min(args.conf, args.conf_defecto), retina_masks=True,
                           verbose=False, **({"device": args.device} if args.device else {}))[0]
        mt = buscar_modelo_multitarea(modelo.predictor.model)
        probs = torch.softmax(mt.madurez_logits.float(), dim=1)[0].cpu().numpy()
        latencias.append(sum(r.speed.values()))

        alto, ancho = r.orig_shape
        pr_p = np.zeros((alto, ancho), bool)
        pr_d = np.zeros((alto, ancho), bool)
        if r.masks is not None:
            mascaras = r.masks.data.cpu().numpy() > 0.5
            clases = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for mk, c, s in zip(mascaras, clases, confs):
                if c == 0 and s >= args.conf:
                    pr_p |= mk
                elif c == 1 and s >= args.conf_defecto:
                    pr_d |= mk
        pr_p = fruto_completo(pr_p, pr_d)  # fruto entero, comparable con el poligono anotado
        pr_d_roi = filtrar_defecto_por_roi(pr_p, pr_d, args.kernel_roi)

        stem = os.path.splitext(os.path.basename(ruta))[0]
        gt_p, gt_d = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), alto, ancho)
        gt_p, gt_d = gt_p.astype(bool), gt_d.astype(bool)

        m_gt = madurez_desde_archivo(ruta)
        m_pr = int(probs.argmax())
        t_gt = torch.from_numpy(np.stack([gt_p, gt_d])[None])
        acc_crudo.actualizar_binario(torch.from_numpy(np.stack([pr_p, pr_d])[None]), t_gt,
                                     torch.tensor([m_pr]), torch.tensor([m_gt]))
        acc_roi.actualizar_binario(torch.from_numpy(np.stack([pr_p, pr_d_roi])[None]), t_gt,
                                   torch.tensor([m_pr]), torch.tensor([m_gt]))

        r_gt, r_pr = calcular_ratio(gt_p, gt_d), calcular_ratio(pr_p, pr_d_roi)
        c_gt, c_pr = clasificar_ocde(r_gt), clasificar_ocde(r_pr)
        ocde_gt.append(c_gt)
        ocde_pr.append(c_pr)
        filas.append({
            "archivo": os.path.basename(ruta), "madurez_real": m_gt + 1, "madurez_pred": m_pr + 1,
            "confianza_madurez": round(float(probs.max()), 4),
            "ratio_real": round(r_gt, 5), "ratio_pred": round(r_pr, 5),
            "ocde_real": CATEGORIAS_OCDE[c_gt], "ocde_pred": CATEGORIAS_OCDE[c_pr],
            "latencia_ms": round(latencias[-1], 2),
        })
        if n < args.n_vis:
            guardar_vis(os.path.join(vis_dir, f"{n:02d}_{stem[:40]}.jpg"), r.orig_img, gt_p, gt_d, pr_p, pr_d_roi,
                        f"REAL  mad {m_gt+1} | {CATEGORIAS_OCDE[c_gt]} ({r_gt*100:.1f}%)",
                        f"PRED  mad {m_pr+1} ({probs.max()*100:.0f}%) | {CATEGORIAS_OCDE[c_pr]} ({r_pr*100:.1f}%)")
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(imagenes)}")

    m_crudo, m_roi = acc_crudo.calcular(), acc_roi.calcular()
    matriz_mad = confusion_matrix(acc_crudo.gt_madurez, acc_crudo.pred_madurez, labels=list(range(5)))
    matriz_ocde = confusion_matrix(ocde_gt, ocde_pr, labels=[0, 1, 2])
    acc_ocde = float(np.trace(matriz_ocde) / max(1, matriz_ocde.sum()))
    mae_ratio = float(np.mean([abs(f["ratio_real"] - f["ratio_pred"]) for f in filas]))
    lat = np.array(latencias[3:] if len(latencias) > 5 else latencias)  # descarta las primeras (calentamiento)

    def seg(m):
        return {k: round(v, 4) for k, v in m.items() if "madurez" not in k and k != "score"}

    resumen = {
        "weights": args.weights, "split": args.split, "n_imagenes": len(filas), "imgsz": args.imgsz,
        "conf_palta": args.conf, "conf_defecto": args.conf_defecto, "kernel_roi": args.kernel_roi,
        "ultralytics": {"global": stats, "por_clase": por_clase},
        "pixel_sin_roi": seg(m_crudo), "pixel_con_roi": seg(m_roi),
        "madurez": {
            "accuracy": round(m_crudo["acc_madurez"], 4), "f1_macro": round(m_crudo["f1_madurez"], 4),
            "matriz_confusion": matriz_mad.tolist(),
            "por_clase": classification_report(acc_crudo.gt_madurez, acc_crudo.pred_madurez, labels=list(range(5)),
                                               target_names=NOMBRES_MADUREZ, zero_division=0, output_dict=True),
        },
        "ocde": {"accuracy": round(acc_ocde, 4), "mae_ratio": round(mae_ratio, 5),
                 "matriz_confusion": matriz_ocde.tolist(), "etiquetas": CATEGORIAS_OCDE},
        "latencia_ms": {"media": round(float(lat.mean()), 2), "p95": round(float(np.percentile(lat, 95)), 2),
                        "dispositivo": str(modelo.predictor.device)},
    }
    with open(os.path.join(out_dir, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    graficar_matriz(matriz_mad, NOMBRES_MADUREZ, f"Madurez ({args.split})", os.path.join(out_dir, "matriz_madurez.png"))
    graficar_matriz(matriz_ocde, CATEGORIAS_OCDE, f"OCDE ({args.split})", os.path.join(out_dir, "matriz_ocde.png"))

    print("\n==== RESUMEN ====")
    print(f"Mascaras (Ultralytics): P {stats.get('metrics/precision(M)')} | R {stats.get('metrics/recall(M)')} | "
          f"F1 {stats.get('metrics/F1(M)')} | mAP50 {stats.get('metrics/mAP50(M)')} | mAP50-95 {stats.get('metrics/mAP50-95(M)')}")
    print(f"Pixel IoU defecto: sin ROI {m_crudo['iou_defecto']:.4f} | con ROI {m_roi['iou_defecto']:.4f}")
    print(f"Madurez: accuracy {m_crudo['acc_madurez']:.4f} | F1 macro {m_crudo['f1_madurez']:.4f}")
    print(f"OCDE: accuracy {acc_ocde:.4f} | MAE ratio {mae_ratio:.4f}")
    print(f"Latencia: {lat.mean():.1f} ms/imagen (p95 {np.percentile(lat, 95):.1f}) en {modelo.predictor.device}")
    print(f"Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
