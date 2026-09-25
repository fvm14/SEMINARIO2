"""
metrics.py
Acumulador de metricas para validacion/test del modelo multitarea.

Segmentacion: IoU, Dice, Precision, Recall por clase (palta, defecto),
calculados a nivel de dataset (suma de TP/FP/FN de todos los pixeles), lo
que evita que imagenes sin defecto distorsionen el promedio.
Madurez: accuracy, F1 macro y matriz de confusion 5x5.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score, confusion_matrix

CLASES_SEG = ["palta", "defecto"]


class AcumuladorMetricas:
    def __init__(self, umbral: float = 0.5, n_clases_madurez: int = 5):
        self.umbral = umbral
        self.n_clases_madurez = n_clases_madurez
        self.reset()

    def reset(self):
        self.tp = np.zeros(2, dtype=np.float64)
        self.fp = np.zeros(2, dtype=np.float64)
        self.fn = np.zeros(2, dtype=np.float64)
        self.pred_madurez = []
        self.gt_madurez = []

    @torch.no_grad()
    def actualizar(self, salida: dict, mascaras_gt: torch.Tensor, madurez_gt: torch.Tensor):
        pred = (torch.sigmoid(salida["seg"].float()) > self.umbral)
        self.actualizar_binario(pred, mascaras_gt > 0.5,
                                salida["madurez"].argmax(dim=1), madurez_gt)

    @torch.no_grad()
    def actualizar_binario(self, pred: torch.Tensor, gt: torch.Tensor,
                           pred_madurez: torch.Tensor, gt_madurez: torch.Tensor):
        """pred, gt: tensores bool (B, 2, H, W). Permite evaluar mascaras ya
        post-procesadas (por ejemplo, con filtro ROI)."""
        dims = (0, 2, 3)
        self.tp += (pred & gt).sum(dim=dims).cpu().numpy()
        self.fp += (pred & ~gt).sum(dim=dims).cpu().numpy()
        self.fn += (~pred & gt).sum(dim=dims).cpu().numpy()
        self.pred_madurez.extend(pred_madurez.cpu().tolist())
        self.gt_madurez.extend(gt_madurez.cpu().tolist())

    def calcular(self) -> dict:
        eps = 1e-9
        res = {}
        for i, nombre in enumerate(CLASES_SEG):
            tp, fp, fn = self.tp[i], self.fp[i], self.fn[i]
            res[f"iou_{nombre}"] = tp / (tp + fp + fn + eps)
            res[f"dice_{nombre}"] = 2 * tp / (2 * tp + fp + fn + eps)
            res[f"precision_{nombre}"] = tp / (tp + fp + eps)
            res[f"recall_{nombre}"] = tp / (tp + fn + eps)

        gt = np.array(self.gt_madurez)
        pred = np.array(self.pred_madurez)
        res["acc_madurez"] = float((gt == pred).mean()) if len(gt) else 0.0
        res["f1_madurez"] = float(f1_score(gt, pred, average="macro", zero_division=0)) if len(gt) else 0.0
        # score combinado para elegir el mejor checkpoint: la palta es facil de
        # segmentar (IoU alto casi siempre), lo dificil es defecto y madurez
        res["score"] = 0.5 * res["iou_defecto"] + 0.5 * res["f1_madurez"]
        return res

    def matriz_confusion(self):
        return confusion_matrix(self.gt_madurez, self.pred_madurez,
                                labels=list(range(self.n_clases_madurez)))
