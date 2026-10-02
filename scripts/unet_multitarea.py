"""
unet_multitarea.py
Modelo de COMPARACION: U-Net con codificador ResNet34 (preentrenado en
ImageNet) y cabezal de madurez. Segmentacion SEMANTICA por pixel (no separa
instancias): dos mascaras independientes, palta y defecto, mas la madurez 1-5.

Se entrena con los mismos datos que el YOLOv8s-seg multitarea de la Ronda 1
(data/yolo_ampliado: 852 anotadas + 4,623 pseudoetiquetas; val y test con
anotacion humana) y con hiperparametros equivalentes: 800 px, lote 8, 100
epocas, paciencia 20, semilla 42, AdamW, peso de madurez 2, sin mezcla de
imagenes y con los mismos aumentos (rotacion 15, traslacion 0.1, escala 0.5,
volteos 0.5, hsv 0.01/0.4/0.3).

Uso (desde la raiz del proyecto, con el entorno activado):
    python scripts/unet_multitarea.py                                   # entrenamiento completo
    python scripts/unet_multitarea.py --epochs 1 --max-batches 5 --name prueba   # prueba rapida
    python scripts/unet_multitarea.py --resume resultados\\comparacion\\unet_resnet34\\last.pt
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import AcumuladorMetricas  # noqa: E402
from rasterize_utils import poligonos_a_mascaras  # noqa: E402

MEDIA = np.array([0.485, 0.456, 0.406], np.float32)
DESV = np.array([0.229, 0.224, 0.225], np.float32)
RELLENO = 114


# ----------------------------------------------------------------- modelo
class Bloque(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.red(x)


class UNetMultitarea(nn.Module):
    """Codificador ResNet34 (strides 2, 4, 8, 16, 32) + decodificador U-Net con
    conexiones de salto + cabezal de madurez sobre el ultimo mapa (512 canales)."""

    def __init__(self, preentrenado=True, n_madurez=5):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if preentrenado else None)
        self.c0 = nn.Sequential(r.conv1, r.bn1, r.relu)   # 64, stride 2
        self.pool = r.maxpool
        self.c1, self.c2, self.c3, self.c4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64, 128, 256, 512
        self.d4 = Bloque(512 + 256, 256)
        self.d3 = Bloque(256 + 128, 128)
        self.d2 = Bloque(128 + 64, 64)
        self.d1 = Bloque(64 + 64, 32)
        self.salida = nn.Conv2d(32, 2, 1)                 # canal 0 palta, canal 1 defecto (stride 2)
        self.madurez = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, 256), nn.SiLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, n_madurez))

    @staticmethod
    def _subir(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        e0 = self.c0(x)
        e1 = self.c1(self.pool(e0))
        e2 = self.c2(e1)
        e3 = self.c3(e2)
        e4 = self.c4(e3)
        y = self.d4(torch.cat([self._subir(e4, e3), e3], 1))
        y = self.d3(torch.cat([self._subir(y, e2), e2], 1))
        y = self.d2(torch.cat([self._subir(y, e1), e1], 1))
        y = self.d1(torch.cat([self._subir(y, e0), e0], 1))
        seg = F.interpolate(self.salida(y), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return {"seg": seg, "madurez": self.madurez(e4)}


# ------------------------------------------------------------------ datos
def madurez_desde_archivo(ruta):
    nivel = int(os.path.splitext(os.path.basename(ruta))[0].split("_")[4].split(".")[0])
    if not 1 <= nivel <= 5:
        raise ValueError(f"Madurez fuera de rango en {ruta}")
    return nivel - 1


def letterbox(img, lado, interp=cv2.INTER_LINEAR, relleno=RELLENO):
    """Ajusta la imagen a lado x lado conservando la proporcion. Devuelve tambien
    (escala, x0, y0) para deshacer el ajuste."""
    h, w = img.shape[:2]
    if h == lado and w == lado:
        return img, (1.0, 0, 0)
    e = min(lado / h, lado / w)
    nh, nw = round(h * e), round(w * e)
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    lienzo = np.full((lado, lado) + img.shape[2:], relleno, img.dtype)
    y0, x0 = (lado - nh) // 2, (lado - nw) // 2
    lienzo[y0:y0 + nh, x0:x0 + nw] = r
    return lienzo, (e, x0, y0)


def normalizar(img_rgb):
    x = (img_rgb.astype(np.float32) / 255.0 - MEDIA) / DESV
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def leer_lista(data_dir, split):
    """Lee <data_dir>/<split>.txt. Solo usa el nombre de cada archivo, asi la
    lista funciona aunque las rutas absolutas sean de otra maquina."""
    with open(os.path.join(data_dir, f"{split}.txt"), encoding="utf-8") as f:
        nombres = [l.strip().replace("\\", "/").split("/")[-1] for l in f if l.strip()]
    return [os.path.join(data_dir, "images", n) for n in nombres]


class DatosPalta(Dataset):
    def __init__(self, data_dir, split, lado=800, aumentar=False):
        self.rutas = leer_lista(data_dir, split)
        self.dir_labels = os.path.join(data_dir, "labels")
        self.lado, self.aumentar = lado, aumentar

    def __len__(self):
        return len(self.rutas)

    def _aumentar(self, img, masc):
        lado = self.lado
        # afin: rotacion +-15, escala 0.5-1.5, traslacion +-0.1 (como el YOLO multitarea)
        ang, esc = random.uniform(-15, 15), random.uniform(0.5, 1.5)
        M = cv2.getRotationMatrix2D((lado / 2, lado / 2), ang, esc)
        M[:, 2] += [random.uniform(-0.1, 0.1) * lado, random.uniform(-0.1, 0.1) * lado]
        img = cv2.warpAffine(img, M, (lado, lado), flags=cv2.INTER_LINEAR, borderValue=(RELLENO,) * 3)
        masc = cv2.warpAffine(masc, M, (lado, lado), flags=cv2.INTER_NEAREST, borderValue=0)
        if random.random() < 0.5:
            img, masc = img[:, ::-1], masc[:, ::-1]
        if random.random() < 0.5:
            img, masc = img[::-1], masc[::-1]
        # color suave (hsv 0.01 / 0.4 / 0.3): la madurez se ve en el color de la cascara
        gh, gs, gv = 1 + np.random.uniform(-1, 1, 3) * [0.01, 0.4, 0.3]
        h, s, v = cv2.split(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV))
        x = np.arange(256, dtype=np.float32)
        hsv = cv2.merge((cv2.LUT(h, ((x * gh) % 180).astype(np.uint8)),
                         cv2.LUT(s, np.clip(x * gs, 0, 255).astype(np.uint8)),
                         cv2.LUT(v, np.clip(x * gv, 0, 255).astype(np.uint8))))
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), np.ascontiguousarray(masc)

    def __getitem__(self, i):
        ruta = self.rutas[i]
        img = cv2.imread(ruta)
        if img is None:
            raise FileNotFoundError(ruta)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        alto, ancho = img.shape[:2]
        stem = os.path.splitext(os.path.basename(ruta))[0]
        palta, defecto = poligonos_a_mascaras(os.path.join(self.dir_labels, stem + ".txt"), alto, ancho)
        masc = np.stack([palta, defecto], -1).astype(np.uint8)
        img, _ = letterbox(img, self.lado)
        masc, _ = letterbox(masc, self.lado, cv2.INTER_NEAREST, 0)
        if self.aumentar:
            img, masc = self._aumentar(img, masc)
        return {"imagen": normalizar(img),
                "mascaras": torch.from_numpy(masc.transpose(2, 0, 1).copy()).float(),
                "madurez": torch.tensor(madurez_desde_archivo(ruta), dtype=torch.long)}


# ---------------------------------------------------------------- perdida
def perdida_multitarea(salida, mascaras, madurez, peso_madurez):
    logits = salida["seg"].float()
    bce = F.binary_cross_entropy_with_logits(logits, mascaras)
    p, t = torch.sigmoid(logits).flatten(2), mascaras.flatten(2)
    dice = 1 - ((2 * (p * t).sum(2) + 1e-6) / (p.sum(2) + t.sum(2) + 1e-6)).mean()
    ce = F.cross_entropy(salida["madurez"].float(), madurez)
    return bce + dice + peso_madurez * ce, {"bce": bce.item(), "dice": dice.item(), "madurez": ce.item()}


def epoca(modelo, loader, device, amp, peso_madurez, opt=None, scaler=None, sched=None, max_batches=0):
    entrenando = opt is not None
    modelo.train(entrenando)
    acum, suma, n = AcumuladorMetricas(), 0.0, 0
    with torch.set_grad_enabled(entrenando):
        for i, lote in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = lote["imagen"].to(device, non_blocking=True)
            m = lote["mascaras"].to(device, non_blocking=True)
            y = lote["madurez"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                salida = modelo(x)
            total, _ = perdida_multitarea(salida, m, y, peso_madurez)
            if entrenando:
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                if (i + 1) % 100 == 0:
                    print(f"    lote {i + 1}/{len(loader)}  perdida {total.item():.3f}", flush=True)
            suma += total.item() * x.shape[0]
            n += x.shape[0]
            acum.actualizar({k: v.float() for k, v in salida.items()}, m, y)
    res = acum.calcular()
    res["perdida"] = suma / max(1, n)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "yolo_ampliado"), help="carpeta con images/, labels/ y train/val.txt")
    p.add_argument("--project", default=os.path.join("resultados", "comparacion"))
    p.add_argument("--name", default="unet_resnet34")
    p.add_argument("--imgsz", type=int, default=800)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3, help="decodificador y cabezal")
    p.add_argument("--lr-encoder", type=float, default=2e-4, help="codificador preentrenado")
    p.add_argument("--peso-madurez", type=float, default=2.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=0, help="solo para pruebas")
    p.add_argument("--sin-preentrenado", action="store_true")
    p.add_argument("--resume", default="")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if amp else "  (SIN GPU: sera muy lento)"))

    ds_tr = DatosPalta(args.data, "train", args.imgsz, aumentar=True)
    ds_va = DatosPalta(args.data, "val", args.imgsz)
    print(f"Train: {len(ds_tr)} imagenes | Val: {len(ds_va)}")
    kw = dict(num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, **kw)

    modelo = UNetMultitarea(preentrenado=not args.sin_preentrenado).to(device)
    print(f"Parametros: {sum(q.numel() for q in modelo.parameters()) / 1e6:.2f} M")
    enc = [q for n, q in modelo.named_parameters() if n.split(".")[0] in ("c0", "c1", "c2", "c3", "c4")]
    resto = [q for n, q in modelo.named_parameters() if n.split(".")[0] not in ("c0", "c1", "c2", "c3", "c4")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr_encoder}, {"params": resto, "lr": args.lr}], weight_decay=1e-4)
    pasos = (min(len(dl_tr), args.max_batches) if args.max_batches else len(dl_tr))
    calent, total = 3 * pasos, args.epochs * pasos
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / calent if s < calent else
                                              0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - calent) / max(1, total - calent)))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    inicio, mejor, sin_mejora = 0, -1.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        modelo.load_state_dict(ck["modelo"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        inicio, mejor, sin_mejora = ck["epoca"] + 1, ck["mejor"], ck["sin_mejora"]
        salida = os.path.dirname(os.path.abspath(args.resume))
        print(f"Reanudando desde la epoca {inicio + 1} (mejor puntaje {mejor:.4f})")
    else:
        salida = os.path.abspath(os.path.join(args.project, args.name))
        if os.path.exists(os.path.join(salida, "last.pt")):
            sys.exit(f"Ya existe {salida}. Usa otro --name o --resume {os.path.join(salida, 'last.pt')}")
        os.makedirs(salida, exist_ok=True)
        with open(os.path.join(salida, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
    print(f"Resultados en: {salida}\n")

    t_ini = time.time()
    for ep in range(inicio, args.epochs):
        t0 = time.time()
        tr = epoca(modelo, dl_tr, device, amp, args.peso_madurez, opt, scaler, sched, args.max_batches)
        va = epoca(modelo, dl_va, device, amp, args.peso_madurez, max_batches=args.max_batches)
        fila = {"epoca": ep + 1, "seg": round(time.time() - t0, 1), "perdida_train": round(tr["perdida"], 4),
                **{f"val_{k}": round(v, 4) for k, v in va.items()}}
        ruta_csv = os.path.join(salida, "results.csv")
        nuevo = not os.path.exists(ruta_csv)
        with open(ruta_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fila))
            if nuevo:
                w.writeheader()
            w.writerow(fila)
        print(f"[{ep + 1}/{args.epochs}] {fila['seg']:.0f}s | perdida {tr['perdida']:.3f} / val {va['perdida']:.3f} | "
              f"IoU palta {va['iou_palta']:.3f} defecto {va['iou_defecto']:.3f} | "
              f"madurez acc {va['acc_madurez']:.3f} F1 {va['f1_madurez']:.3f} | puntaje {va['score']:.4f}", flush=True)
        estado = {"modelo": modelo.state_dict(), "epoca": ep, "config": vars(args), "metricas_val": va}
        if va["score"] > mejor:
            mejor, sin_mejora = va["score"], 0
            torch.save(estado, os.path.join(salida, "best.pt"))
            print("   -> nuevo mejor modelo")
        else:
            sin_mejora += 1
        torch.save({**estado, "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "mejor": mejor, "sin_mejora": sin_mejora}, os.path.join(salida, "last.pt"))
        if sin_mejora >= args.patience:
            print(f"Parada temprana: {args.patience} epocas sin mejora.")
            break
    print(f"\nListo en {(time.time() - t_ini) / 3600:.2f} h. Mejor puntaje de validacion: {mejor:.4f}")
    print(f"Pesos: {os.path.join(salida, 'best.pt')}")


if __name__ == "__main__":
    main()
