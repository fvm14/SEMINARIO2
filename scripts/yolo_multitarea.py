"""
yolo_multitarea.py
YOLOv8s-seg extendido con un cabezal de clasificacion de madurez (1-5).

Arquitectura:
  backbone YOLOv8s (capas 0-9, termina en SPPF, stride 32)
     |-- neck + cabezal Segment de YOLO  -> deteccion + mascaras (palta, defecto)
     |-- cabezal de madurez (nuevo)       -> 5 logits
  El backbone es compartido: el gradiente de la perdida de madurez tambien
  actualiza el backbone, por eso es un modelo multitarea real y no dos modelos.

Integracion con Ultralytics (probado con ultralytics==8.4.160):
  - YoloMultitarea: el forward de YOLO no cambia (asi el validador, el
    predictor y el exportador siguen funcionando). Los logits de madurez se
    guardan en `self.madurez_logits` en cada forward.
  - loss(): suma la perdida de YOLO + peso * CrossEntropy(madurez) y agrega
    "mad_loss" al diccionario de perdidas que Ultralytics registra.
  - La etiqueta de madurez se lee del nombre del archivo (batch["im_file"]),
    igual que motor_ocde.extraer_madurez: T10_d01_033_a_<madurez>_...
  - ValidadorMultitarea: agrega metrics/madurez_acc y metrics/madurez_f1 y
    redefine fitness = 0.5 * fitness_mascaras + 0.5 * F1_madurez para elegir
    best.pt considerando ambas tareas.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import LOGGER, RANK

N_MADUREZ = 5
CAPA_MADUREZ = 9  # SPPF, ultima capa del backbone YOLOv8


def madurez_desde_archivo(ruta: str) -> int:
    """Devuelve la madurez 0..4 (para CrossEntropy) a partir del nombre del archivo."""
    # sirve para nombres originales (T10_d03_458_b_1.jpg) y de Roboflow (T10_d03_458_b_1_jpg.rf.xxx.jpg)
    base = os.path.splitext(os.path.basename(ruta))[0]
    partes = base.split("_")
    nivel = int(partes[4].split(".")[0])
    if not 1 <= nivel <= N_MADUREZ:
        raise ValueError(f"Nivel de madurez fuera de rango en {ruta}")
    return nivel - 1


class CabezalMadurez(nn.Module):
    def __init__(self, canales_in: int, n_clases: int = N_MADUREZ, dropout: float = 0.2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(canales_in, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_clases),
        )

    def forward(self, x):
        return self.mlp(self.pool(x))


class YoloMultitarea(SegmentationModel):
    def __init__(self, cfg="yolov8s-seg.yaml", ch=3, nc=None, verbose=True,
                 peso_madurez: float = 1.0):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        capa = self.model[CAPA_MADUREZ]
        canales = [m for m in capa.modules() if isinstance(m, nn.Conv2d)][-1].out_channels
        self.cabezal_madurez = CabezalMadurez(canales)
        self.peso_madurez = peso_madurez
        self.madurez_logits = None

    def __getstate__(self):
        # no guardar en el checkpoint los logits del ultimo forward
        estado = self.__dict__.copy()
        estado["madurez_logits"] = None
        return estado

    def _predict_once(self, x, profile=False, embed=None):
        # Igual que BaseModel._predict_once (ultralytics 8.4) + captura de la capa 9
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i == CAPA_MADUREZ and "cabezal_madurez" in self._modules:
                # (durante el __init__ de YOLO se hace un forward antes de crear el cabezal)
                self.madurez_logits = self.cabezal_madurez(x)
            if m.i in embed:
                embeddings.append(F.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        if preds is None:
            preds = self.forward(batch["img"])
        perdida, items = super().loss(batch, preds)

        objetivo = torch.tensor([madurez_desde_archivo(f) for f in batch["im_file"]],
                                device=self.madurez_logits.device, dtype=torch.long)
        ce = F.cross_entropy(self.madurez_logits.float(), objetivo)
        n = objetivo.shape[0]
        # Ultralytics multiplica cada componente por el tamano de batch; se replica la escala
        perdida = torch.cat([perdida, (self.peso_madurez * ce * n).reshape(1).to(perdida.dtype)])
        items = {**items, "mad_loss": ce.detach()}
        return perdida, items


def buscar_modelo_multitarea(modelo):
    """Durante el entrenamiento el validador recibe el modelo directo; en
    validacion independiente recibe un AutoBackend que lo envuelve."""
    actual = modelo
    for _ in range(3):
        if isinstance(actual, YoloMultitarea):
            return actual
        actual = getattr(actual, "model", None)
        if actual is None:
            break
    raise TypeError("El modelo no es YoloMultitarea (no tiene cabezal de madurez).")


class ValidadorMultitarea(SegmentationValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        self._modelo_mt = buscar_modelo_multitarea(model)
        self.madurez_pred, self.madurez_gt = [], []

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        logits = self._modelo_mt.madurez_logits
        self.madurez_pred.extend(logits.argmax(dim=1).cpu().tolist())
        self.madurez_gt.extend(madurez_desde_archivo(f) for f in batch["im_file"])

    def get_stats(self):
        stats = super().get_stats()
        gt, pred = np.array(self.madurez_gt), np.array(self.madurez_pred)
        acc = float((gt == pred).mean()) if len(gt) else 0.0
        f1 = float(f1_score(gt, pred, average="macro", zero_division=0)) if len(gt) else 0.0
        stats["metrics/madurez_acc"] = acc
        stats["metrics/madurez_f1"] = f1
        fit_mascaras = 0.1 * stats.get("metrics/mAP50(M)", 0.0) + 0.9 * stats.get("metrics/mAP50-95(M)", 0.0)
        stats["fitness"] = 0.5 * fit_mascaras + 0.5 * f1
        self.matriz_madurez = confusion_matrix(gt, pred, labels=list(range(N_MADUREZ)))
        return stats

    def print_results(self):
        super().print_results()
        gt, pred = np.array(self.madurez_gt), np.array(self.madurez_pred)
        if len(gt):
            acc = (gt == pred).mean()
            f1 = f1_score(gt, pred, average="macro", zero_division=0)
            LOGGER.info(f"Madurez: accuracy {acc:.4f} | F1 macro {f1:.4f} | n={len(gt)}")
            LOGGER.info("Matriz de confusion madurez (filas=real 1..5, columnas=pred 1..5):\n"
                        f"{confusion_matrix(gt, pred, labels=list(range(N_MADUREZ)))}")


class TrainerMultitarea(SegmentationTrainer):
    peso_madurez = 1.0  # se asigna desde train_yolo_multitarea.py antes de instanciar

    def get_model(self, cfg=None, weights=None, verbose=True):
        modelo = YoloMultitarea(cfg, nc=self.data["nc"], ch=self.data["channels"],
                                verbose=verbose and RANK == -1, peso_madurez=self.peso_madurez)
        modelo = self.set_model_names_for_load(modelo)
        if weights:
            modelo.load(weights)  # transfiere pesos COCO de YOLOv8s-seg; el cabezal de madurez queda aleatorio
        return modelo

    def get_validator(self):
        from copy import copy
        return ValidadorMultitarea(self.test_loader, save_dir=self.save_dir,
                                   args=copy(self.args), _callbacks=self.callbacks)
