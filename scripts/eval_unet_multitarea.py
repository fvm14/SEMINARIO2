"""
eval_unet_multitarea.py
Evalua la U-Net ResNet34 multitarea con el MISMO protocolo que el YOLOv8s-seg
multitarea (eval_yolo_multitarea.py): mascaras a la resolucion original,
fruto completo, filtro ROI de 20 px, ratio defecto/fruto, categoria OCDE,
madurez, y los mismos archivos de salida (metricas.json y predicciones.csv),
de modo que bootstrap_ic.py sirve para comparar ambos modelos.

1) Calibrar el umbral de defecto en VALIDACION (max IoU de defecto):
    python scripts/eval_unet_multitarea.py --weights resultados\\comparacion\\unet_resnet34\\best.pt --calibrar
2) Evaluar en TEST con ese umbral:
    python scripts/eval_unet_multitarea.py --weights resultados\\comparacion\\unet_resnet34\\best.pt --umbral-defecto 0.5
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import AcumuladorMetricas  # noqa: E402
from ocde import (CATEGORIAS_OCDE, KERNEL_DILATACION_ROI, calcular_ratio, clasificar_ocde,  # noqa: E402
                  filtrar_defecto_por_roi, fruto_completo)
from rasterize_utils import poligonos_a_mascaras  # noqa: E402
from unet_multitarea import UNetMultitarea, leer_lista, letterbox, madurez_desde_archivo, normalizar  # noqa: E402

UMBRALES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@torch.no_grad()
def predecir(modelo, img_bgr, lado, device):
    """Devuelve (prob_palta, prob_defecto) a la resolucion original, las
    probabilidades de madurez y el tiempo de inferencia en ms."""
    alto, ancho = img_bgr.shape[:2]
    x, (e, x0, y0) = letterbox(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), lado)
    t = normalizar(x)[None].to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    salida = modelo(t)
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    prob = torch.sigmoid(salida["seg"].float())[0].cpu().numpy()
    nh, nw = round(alto * e), round(ancho * e)
    prob = prob[:, y0:y0 + nh, x0:x0 + nw]
    if (nh, nw) != (alto, ancho):
        prob = np.stack([cv2.resize(c, (ancho, alto), interpolation=cv2.INTER_LINEAR) for c in prob])
    mad = torch.softmax(salida["madurez"].float(), 1)[0].cpu().numpy()
    return prob[0], prob[1], mad, ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default=os.path.join("data", "yolo"), help="carpeta con images/ y val/test.txt (solo anotacion humana)")
    p.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    p.add_argument("--split", default="", help="por defecto: val si --calibrar, si no test")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--umbral-palta", type=float, default=0.5)
    p.add_argument("--umbral-defecto", type=float, default=0.5)
    p.add_argument("--calibrar", action="store_true", help="barre el umbral de defecto en validacion")
    p.add_argument("--kernel-roi", type=int, default=KERNEL_DILATACION_ROI)
    p.add_argument("--device", default="")
    p.add_argument("--max-imagenes", type=int, default=0)
    p.add_argument("--out-dir", default="")
    args = p.parse_args()
    split = args.split or ("val" if args.calibrar else "test")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    modelo = UNetMultitarea(preentrenado=False).to(device).eval()
    modelo.load_state_dict(ck["modelo"])
    n_param = sum(q.numel() for q in modelo.parameters())
    rutas = leer_lista(args.data, split)
    if args.max_imagenes:
        rutas = rutas[:args.max_imagenes]
    print(f"Modelo: {n_param / 1e6:.2f} M parametros | {split}: {len(rutas)} imagenes | {device}")

    umbrales = UMBRALES if args.calibrar else [args.umbral_defecto]
    cuenta = {u: dict(tp=0, fp=0, fn=0, ok=0, rech=0, rech_ok=0, mae=0.0) for u in umbrales}
    acc_crudo, acc_roi = AcumuladorMetricas(), AcumuladorMetricas()
    filas, ocde_gt, ocde_pr, lat = [], [], [], []

    for n, ruta in enumerate(rutas):
        img = cv2.imread(ruta)
        alto, ancho = img.shape[:2]
        pp, pd, mad, ms = predecir(modelo, img, args.imgsz, device)
        lat.append(ms)
        stem = os.path.splitext(os.path.basename(ruta))[0]
        gt_p, gt_d = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), alto, ancho)
        gt_p, gt_d = gt_p.astype(bool), gt_d.astype(bool)
        r_gt = calcular_ratio(gt_p, gt_d)
        c_gt = clasificar_ocde(r_gt)
        palta = pp > args.umbral_palta
        for u in umbrales:
            defecto = pd > u
            fruto = fruto_completo(palta, defecto)
            d_roi = filtrar_defecto_por_roi(fruto, defecto, args.kernel_roi)
            r_pr = calcular_ratio(fruto, d_roi)
            c_pr = clasificar_ocde(r_pr)
            c = cuenta[u]
            c["tp"] += int((d_roi & gt_d).sum()); c["fp"] += int((d_roi & ~gt_d).sum()); c["fn"] += int((~d_roi & gt_d).sum())
            c["ok"] += int(c_pr == c_gt); c["rech"] += int(c_gt == 2); c["rech_ok"] += int(c_gt == 2 and c_pr == 2)
            c["mae"] += abs(r_gt - r_pr)
        if not args.calibrar:
            m_gt, m_pr = madurez_desde_archivo(ruta), int(mad.argmax())
            t_gt = torch.from_numpy(np.stack([gt_p, gt_d])[None])
            acc_crudo.actualizar_binario(torch.from_numpy(np.stack([fruto, defecto])[None]), t_gt, torch.tensor([m_pr]), torch.tensor([m_gt]))
            acc_roi.actualizar_binario(torch.from_numpy(np.stack([fruto, d_roi])[None]), t_gt, torch.tensor([m_pr]), torch.tensor([m_gt]))
            ocde_gt.append(c_gt); ocde_pr.append(c_pr)
            filas.append({"archivo": os.path.basename(ruta), "madurez_real": m_gt + 1, "madurez_pred": m_pr + 1,
                          "confianza_madurez": round(float(mad.max()), 4), "ratio_real": round(r_gt, 5),
                          "ratio_pred": round(r_pr, 5), "ocde_real": CATEGORIAS_OCDE[c_gt],
                          "ocde_pred": CATEGORIAS_OCDE[c_pr], "latencia_ms": round(ms, 2)})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(rutas)}", flush=True)

    N = len(rutas)
    if args.calibrar:
        print(f"\n umbral   iou_defecto  precision  recall  acc_ocde  recall_rechazado  mae_ratio")
        mejor = None
        for u in umbrales:
            c = cuenta[u]
            iou = c["tp"] / max(1, c["tp"] + c["fp"] + c["fn"])
            print(f"  {u:.2f}     {iou:.4f}       {c['tp'] / max(1, c['tp'] + c['fp']):.4f}     {c['tp'] / max(1, c['tp'] + c['fn']):.4f}"
                  f"   {c['ok'] / N:.4f}    {c['rech_ok'] / max(1, c['rech']):.4f}            {c['mae'] / N:.4f}")
            if mejor is None or iou > mejor[1]:
                mejor = (u, iou)
        print(f"\nUmbral recomendado para defecto (max IoU de defecto, {N} imagenes de {split}): {mejor[0]}")
        return

    out_dir = os.path.abspath(args.out_dir or os.path.join(os.path.dirname(args.weights),
                                                           f"eval_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(out_dir, exist_ok=True)
    m_crudo, m_roi = acc_crudo.calcular(), acc_roi.calcular()
    matriz_ocde = confusion_matrix(ocde_gt, ocde_pr, labels=[0, 1, 2])
    c = cuenta[args.umbral_defecto]
    lat_a = np.array(lat[3:] if len(lat) > 5 else lat)
    seg = lambda m: {k: round(v, 4) for k, v in m.items() if "madurez" not in k and k != "score"}  # noqa: E731
    resumen = {
        "weights": args.weights, "modelo": "U-Net ResNet34 multitarea", "parametros_M": round(n_param / 1e6, 2),
        "tamano_mb": round(n_param * 4 / 1e6, 1), "split": split, "n_imagenes": N, "imgsz": args.imgsz,
        "umbral_palta": args.umbral_palta, "umbral_defecto": args.umbral_defecto,
        "pixel_sin_roi": seg(m_crudo), "pixel_con_roi": seg(m_roi),
        "madurez": {"accuracy": round(m_crudo["acc_madurez"], 4), "f1_macro": round(m_crudo["f1_madurez"], 4),
                    "matriz_confusion": acc_crudo.matriz_confusion().tolist(),
                    "por_clase": classification_report(acc_crudo.gt_madurez, acc_crudo.pred_madurez, labels=list(range(5)),
                                                       zero_division=0, output_dict=True)},
        "ocde": {"accuracy": round(c["ok"] / N, 4), "recall_rechazado": round(c["rech_ok"] / max(1, c["rech"]), 4),
                 "mae_ratio": round(c["mae"] / N, 5), "matriz_confusion": matriz_ocde.tolist(), "etiquetas": CATEGORIAS_OCDE},
        "latencia_ms": {"media": round(float(lat_a.mean()), 2), "p95": round(float(np.percentile(lat_a, 95)), 2),
                        "dispositivo": str(device), "hilos_cpu": torch.get_num_threads()},
    }
    with open(os.path.join(out_dir, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    print("\n==== RESUMEN ====")
    print(f"Parametros: {resumen['parametros_M']} M | tamano aprox. {resumen['tamano_mb']} MB (FP32)")
    print(f"Pixel IoU palta {m_roi['iou_palta']:.4f} | defecto {m_roi['iou_defecto']:.4f} "
          f"(precision {m_roi['precision_defecto']:.4f}, recall {m_roi['recall_defecto']:.4f})")
    print(f"Madurez: accuracy {m_crudo['acc_madurez']:.4f} | F1 macro {m_crudo['f1_madurez']:.4f}")
    print(f"OCDE: accuracy {resumen['ocde']['accuracy']:.4f} | recall Rechazado {resumen['ocde']['recall_rechazado']:.4f} | "
          f"MAE ratio {resumen['ocde']['mae_ratio']:.4f}")
    print(f"Latencia de inferencia: {lat_a.mean():.1f} ms/imagen (p95 {np.percentile(lat_a, 95):.1f}) en {device}")
    print(f"Resultados en: {out_dir}")


if __name__ == "__main__":
    main()
