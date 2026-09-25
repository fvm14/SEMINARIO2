"""
evaluate.py
Evalua un checkpoint del modelo multitarea sobre val o test.

Reporta:
  - Segmentacion (palta, defecto): IoU, Dice, Precision, Recall
    * sin post-proceso
    * con post-proceso (palta = mayor componente, defecto filtrado por ROI
      dilatado 20 px, como en evaluar_con_roi.py de Seminario I)
  - Madurez: accuracy, F1 macro, matriz de confusion, reporte por clase
  - OCDE: categoria predicha (a partir de las mascaras del modelo) vs
    categoria ground truth (a partir de los poligonos anotados)
  - Latencia de inferencia (ms por imagen, batch 1)

Uso (desde la raiz del proyecto):
    python scripts/evaluate.py --checkpoint resultados/runs/<run>/best.pt
    python scripts/evaluate.py --checkpoint ... --split val --n-vis 20
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # modulos compartidos en scripts/


import argparse
import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import PaltaMultitaskDataset
from transforms import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import PaltaMultitaskModel
from metrics import AcumuladorMetricas
from ocde import (calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi,
                  mayor_componente, CATEGORIAS_OCDE, KERNEL_DILATACION_ROI)

NOMBRES_MADUREZ = ["1", "2", "3", "4", "5"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--manifest", default=os.path.join("data", "splits", "split_manifest.csv"))
    p.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    p.add_argument("--img-size", type=int, default=0, help="0 = el usado en entrenamiento")
    p.add_argument("--umbral", type=float, default=0.5)
    p.add_argument("--kernel-roi", type=int, default=KERNEL_DILATACION_ROI)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--n-vis", type=int, default=12, help="imagenes a visualizar")
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--out-dir", default="")
    return p.parse_args()


def graficar_matriz(matriz, etiquetas, titulo, ruta):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=30, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo)
    umbral_color = matriz.max() / 2 if matriz.max() else 0
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            ax.text(j, i, int(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > umbral_color else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def desnormalizar(imagen_t):
    img = imagen_t.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def overlay(img_rgb, palta, defecto):
    out = img_rgb.astype(np.float32).copy()
    out[palta] = out[palta] * 0.6 + np.array([0, 200, 0]) * 0.4
    out[defecto] = out[defecto] * 0.4 + np.array([230, 0, 0]) * 0.6
    return out.astype(np.uint8)


def guardar_visualizacion(ruta, img_rgb, gt_p, gt_d, pr_p, pr_d, texto_gt, texto_pr):
    izq = overlay(img_rgb, gt_p, gt_d)
    der = overlay(img_rgb, pr_p, pr_d)
    for panel, texto in [(izq, texto_gt), (der, texto_pr)]:
        cv2.putText(panel, texto, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(panel, texto, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    lienzo = np.concatenate([izq, np.full((izq.shape[0], 6, 3), 255, np.uint8), der], axis=1)
    cv2.imwrite(ruta, cv2.cvtColor(lienzo, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def medir_latencia(modelo, device, img_size, n=50):
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(10):
        modelo(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        modelo(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    img_size = args.img_size or ckpt.get("config", {}).get("img_size", 512)
    modelo = PaltaMultitaskModel(pretrained=False).to(device)
    modelo.load_state_dict(ckpt["modelo"])
    modelo.eval()

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(args.checkpoint), f"eval_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    vis_dir = os.path.join(out_dir, "visualizaciones")
    os.makedirs(vis_dir, exist_ok=True)

    ds = PaltaMultitaskDataset(args.manifest, args.raw_dir, args.split, get_val_transforms(img_size))
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    print(f"Evaluando {args.checkpoint} en '{args.split}' ({len(ds)} imagenes, {img_size}px, {device})")

    acc_crudo = AcumuladorMetricas(umbral=args.umbral)
    acc_post = AcumuladorMetricas(umbral=args.umbral)
    filas = []
    ocde_gt, ocde_pred = [], []
    n_vis = 0

    for i, lote in enumerate(dl):
        if args.max_batches and i >= args.max_batches:
            break
        imagenes = lote["imagen"].to(device)
        mascaras = lote["mascaras"].to(device)
        madurez = lote["madurez"].to(device)

        salida = modelo(imagenes)
        salida = {k: v.float() for k, v in salida.items()}
        acc_crudo.actualizar(salida, mascaras, madurez)

        probs_seg = torch.sigmoid(salida["seg"]).cpu().numpy()
        probs_mad = torch.softmax(salida["madurez"], dim=1).cpu().numpy()
        gt_np = mascaras.cpu().numpy() > 0.5

        post = np.zeros_like(gt_np)
        for b in range(imagenes.shape[0]):
            pr_p = mayor_componente(probs_seg[b, 0] > args.umbral)
            pr_d = filtrar_defecto_por_roi(pr_p, probs_seg[b, 1] > args.umbral, args.kernel_roi)
            post[b, 0], post[b, 1] = pr_p, pr_d
            gt_p, gt_d = gt_np[b, 0], gt_np[b, 1]

            r_gt, r_pr = calcular_ratio(gt_p, gt_d), calcular_ratio(pr_p, pr_d)
            c_gt, c_pr = clasificar_ocde(r_gt), clasificar_ocde(r_pr)
            ocde_gt.append(c_gt)
            ocde_pred.append(c_pr)

            m_gt = int(madurez[b].item())
            m_pr = int(probs_mad[b].argmax())
            filas.append({
                "archivo": lote["archivo"][b],
                "madurez_real": m_gt + 1, "madurez_pred": m_pr + 1,
                "confianza_madurez": round(float(probs_mad[b].max()), 4),
                "ratio_real": round(r_gt, 5), "ratio_pred": round(r_pr, 5),
                "ocde_real": CATEGORIAS_OCDE[c_gt], "ocde_pred": CATEGORIAS_OCDE[c_pr],
            })

            if n_vis < args.n_vis:
                img_rgb = desnormalizar(imagenes[b])
                nombre = os.path.splitext(lote["archivo"][b])[0][:40]
                guardar_visualizacion(
                    os.path.join(vis_dir, f"{n_vis:02d}_{nombre}.jpg"), img_rgb, gt_p, gt_d, pr_p, pr_d,
                    f"REAL  mad {m_gt+1} | {CATEGORIAS_OCDE[c_gt]} ({r_gt*100:.1f}%)",
                    f"PRED  mad {m_pr+1} | {CATEGORIAS_OCDE[c_pr]} ({r_pr*100:.1f}%)")
                n_vis += 1

        acc_post.actualizar_binario(torch.from_numpy(post), torch.from_numpy(gt_np),
                                    salida["madurez"].argmax(dim=1), madurez)

    m_crudo, m_post = acc_crudo.calcular(), acc_post.calcular()
    matriz_mad = acc_crudo.matriz_confusion()
    matriz_ocde = confusion_matrix(ocde_gt, ocde_pred, labels=[0, 1, 2])
    acc_ocde = float(np.trace(matriz_ocde) / max(1, matriz_ocde.sum()))
    mae_ratio = float(np.mean([abs(f["ratio_real"] - f["ratio_pred"]) for f in filas]))
    latencia = medir_latencia(modelo, device, img_size)

    reporte_mad = classification_report(acc_crudo.gt_madurez, acc_crudo.pred_madurez,
                                        labels=list(range(5)), target_names=NOMBRES_MADUREZ,
                                        zero_division=0, output_dict=True)

    resumen = {
        "checkpoint": args.checkpoint, "split": args.split, "n_imagenes": len(filas),
        "img_size": img_size, "dispositivo": str(device),
        "segmentacion_sin_postproceso": {k: round(v, 4) for k, v in m_crudo.items() if "madurez" not in k and k != "score"},
        "segmentacion_con_postproceso_roi": {k: round(v, 4) for k, v in m_post.items() if "madurez" not in k and k != "score"},
        "madurez": {"accuracy": round(m_crudo["acc_madurez"], 4), "f1_macro": round(m_crudo["f1_madurez"], 4),
                    "por_clase": reporte_mad, "matriz_confusion": matriz_mad.tolist()},
        "ocde": {"accuracy": round(acc_ocde, 4), "mae_ratio": round(mae_ratio, 5),
                 "matriz_confusion": matriz_ocde.tolist(), "etiquetas": CATEGORIAS_OCDE},
        "latencia_ms_por_imagen": round(latencia, 2),
    }
    with open(os.path.join(out_dir, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)
    graficar_matriz(matriz_mad, NOMBRES_MADUREZ, f"Madurez ({args.split})", os.path.join(out_dir, "matriz_madurez.png"))
    graficar_matriz(matriz_ocde, CATEGORIAS_OCDE, f"OCDE ({args.split})", os.path.join(out_dir, "matriz_ocde.png"))

    print("\n==== RESULTADOS ====")
    for nombre, m in [("sin post-proceso", m_crudo), ("con post-proceso ROI", m_post)]:
        print(f"Segmentacion {nombre}: IoU palta {m['iou_palta']:.4f} | IoU defecto {m['iou_defecto']:.4f} | "
              f"P defecto {m['precision_defecto']:.4f} | R defecto {m['recall_defecto']:.4f}")
    print(f"Madurez: accuracy {m_crudo['acc_madurez']:.4f} | F1 macro {m_crudo['f1_madurez']:.4f}")
    print(f"OCDE: accuracy {acc_ocde:.4f} | MAE ratio {mae_ratio:.4f}")
    print(f"Latencia: {latencia:.1f} ms/imagen ({device})")
    print(f"Resultados guardados en: {out_dir}")


if __name__ == "__main__":
    main()
