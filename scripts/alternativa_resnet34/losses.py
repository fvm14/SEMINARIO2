"""
losses.py
Perdida combinada para el modelo multitarea:
  - Segmentacion (palta, defecto): BCE + Dice por canal (mascaras
    independientes, no mutuamente excluyentes -> sigmoid, no softmax)
  - Madurez: CrossEntropy estandar (5 clases mutuamente excluyentes)

La perdida total es una suma ponderada; los pesos son ajustables porque las
dos tareas tienen escalas de perdida distintas y no hay garantia de que
converjan al mismo ritmo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """logits, targets: (B, C, H, W). targets en {0,1}. Dice por canal, promediado."""
    probs = torch.sigmoid(logits)
    probs = probs.flatten(2)   # (B, C, H*W)
    targets = targets.flatten(2)

    interseccion = (probs * targets).sum(dim=2)
    union = probs.sum(dim=2) + targets.sum(dim=2)
    dice = (2 * interseccion + eps) / (union + eps)
    return 1 - dice.mean()


class PerdidaMultitarea(nn.Module):
    def __init__(self, peso_bce: float = 1.0, peso_dice: float = 1.0,
                 peso_madurez: float = 1.0):
        super().__init__()
        self.peso_bce = peso_bce
        self.peso_dice = peso_dice
        self.peso_madurez = peso_madurez
        self.bce = nn.BCEWithLogitsLoss()
        self.ce_madurez = nn.CrossEntropyLoss()

    def forward(self, salida: dict, mascaras_gt: torch.Tensor, madurez_gt: torch.Tensor):
        logits_seg = salida["seg"]        # (B, 2, H, W)
        logits_madurez = salida["madurez"]  # (B, 5)

        perdida_bce = self.bce(logits_seg, mascaras_gt)
        perdida_dice = dice_loss(logits_seg, mascaras_gt)
        perdida_seg = self.peso_bce * perdida_bce + self.peso_dice * perdida_dice

        perdida_madurez = self.ce_madurez(logits_madurez, madurez_gt)

        perdida_total = perdida_seg + self.peso_madurez * perdida_madurez

        return {
            "total": perdida_total,
            "bce": perdida_bce.detach(),
            "dice": perdida_dice.detach(),
            "madurez": perdida_madurez.detach(),
        }


if __name__ == "__main__":
    criterio = PerdidaMultitarea()
    logits_seg = torch.randn(2, 2, 64, 64)
    mascaras = (torch.rand(2, 2, 64, 64) > 0.7).float()
    logits_madurez = torch.randn(2, 5)
    madurez = torch.randint(0, 5, (2,))

    salida = {"seg": logits_seg, "madurez": logits_madurez}
    resultado = criterio(salida, mascaras, madurez)
    for k, v in resultado.items():
        print(k, v.item())
