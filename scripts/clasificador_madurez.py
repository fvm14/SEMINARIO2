"""
clasificador_madurez.py
Modelo de COMPARACION (alternativa de dos redes de tarea unica): clasificador
ResNet-34 de madurez (5 niveles), que se combina con el YOLOv8s-seg SIN cabezal
de madurez (yolov8s_base_sem1) para entregar las dos salidas con dos modelos.

Mismos datos que los demas modelos: entrenamiento con data/yolo_ampliado
(5,475 imagenes; la madurez es real en todas, viene del nombre de archivo),
validacion y prueba con data/yolo. Semilla 42, sin mezcla de imagenes, color suave.

1) Entrenar:
    python scripts/clasificador_madurez.py
2) Evaluar en test y combinar con la evaluacion del segmentador (carpeta eval_test_... de yolov8s_base_sem1):
    python scripts/clasificador_madurez.py --evaluar --weights resultados\\comparacion\\resnet34_madurez\\best.pt ^
        --eval-segmentador resultados\\yolo_multitarea\\yolov8s_base_sem1\\eval_test_20261001_192937
   Crea resultados\\comparacion\\dos_redes\\ con predicciones.csv y metricas.json (mismo formato que los otros modelos).
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
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet34_Weights, resnet34

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_multitarea import RELLENO, leer_lista, letterbox, madurez_desde_archivo, normalizar  # noqa: E402


def crear_modelo(preentrenado=True):
    m = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if preentrenado else None)
    m.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 5))
    return m


class DatosMadurez(Dataset):
    def __init__(self, data_dir, split, lado, aumentar=False):
        self.rutas, self.lado, self.aumentar = leer_lista(data_dir, split), lado, aumentar

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        img = cv2.imread(self.rutas[i])
        if img is None:
            raise FileNotFoundError(self.rutas[i])
        img, _ = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), self.lado)
        if self.aumentar:
            lado = self.lado
            M = cv2.getRotationMatrix2D((lado / 2, lado / 2), random.uniform(-15, 15), random.uniform(0.8, 1.2))
            M[:, 2] += [random.uniform(-0.1, 0.1) * lado, random.uniform(-0.1, 0.1) * lado]
            img = cv2.warpAffine(img, M, (lado, lado), flags=cv2.INTER_LINEAR, borderValue=(RELLENO,) * 3)
            if random.random() < 0.5:
                img = img[:, ::-1]
            if random.random() < 0.5:
                img = img[::-1]
            gh, gs, gv = 1 + np.random.uniform(-1, 1, 3) * [0.01, 0.4, 0.3]
            h, s, v = cv2.split(cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV))
            x = np.arange(256, dtype=np.float32)
            img = cv2.cvtColor(cv2.merge((cv2.LUT(h, ((x * gh) % 180).astype(np.uint8)),
                                          cv2.LUT(s, np.clip(x * gs, 0, 255).astype(np.uint8)),
                                          cv2.LUT(v, np.clip(x * gv, 0, 255).astype(np.uint8)))), cv2.COLOR_HSV2RGB)
        return normalizar(img), madurez_desde_archivo(self.rutas[i])


def pasar(modelo, loader, device, amp, opt=None, scaler=None, sched=None, max_batches=0):
    entrenando = opt is not None
    modelo.train(entrenando)
    suma, n, reales, preds = 0.0, 0, [], []
    with torch.set_grad_enabled(entrenando):
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = modelo(x)
            perdida = F.cross_entropy(logits.float(), y)
            if entrenando:
                opt.zero_grad(set_to_none=True)
                scaler.scale(perdida).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            suma += perdida.item() * len(y)
            n += len(y)
            reales += y.tolist()
            preds += logits.argmax(1).tolist()
    r, p = np.array(reales), np.array(preds)
    return suma / max(1, n), float((r == p).mean()), float(f1_score(r, p, labels=list(range(5)), average="macro", zero_division=0))


def entrenar(args, device):
    amp = device.type == "cuda"
    ds_tr = DatosMadurez(args.data, "train", args.imgsz, aumentar=True)
    ds_va = DatosMadurez(args.data_eval, "val", args.imgsz)
    print(f"Train: {len(ds_tr)} imagenes | Val: {len(ds_va)}")
    kw = dict(num_workers=args.workers, pin_memory=amp, persistent_workers=args.workers > 0)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, **kw)
    modelo = crear_modelo(not args.sin_preentrenado).to(device)
    print(f"Parametros: {sum(q.numel() for q in modelo.parameters()) / 1e6:.2f} M")
    opt = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=1e-4)
    pasos = min(len(dl_tr), args.max_batches) if args.max_batches else len(dl_tr)
    calent, total = 2 * pasos, args.epochs * pasos
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / calent if s < calent else
                                              0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - calent) / max(1, total - calent)))))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    salida = os.path.abspath(os.path.join(args.project, args.name))
    if os.path.exists(os.path.join(salida, "best.pt")):
        sys.exit(f"Ya existe {salida}. Usa otro --name.")
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Resultados en: {salida}\n")
    mejor, sin_mejora, t_ini = -1.0, 0, time.time()
    for ep in range(args.epochs):
        t0 = time.time()
        p_tr, _, _ = pasar(modelo, dl_tr, device, amp, opt, scaler, sched, args.max_batches)
        p_va, acc, f1 = pasar(modelo, dl_va, device, amp, max_batches=args.max_batches)
        with open(os.path.join(salida, "results.csv"), "a", newline="") as f:
            w = csv.writer(f)
            if ep == 0:
                w.writerow(["epoca", "seg", "perdida_train", "perdida_val", "val_acc", "val_f1"])
            w.writerow([ep + 1, round(time.time() - t0, 1), round(p_tr, 4), round(p_va, 4), round(acc, 4), round(f1, 4)])
        print(f"[{ep + 1}/{args.epochs}] {time.time() - t0:.0f}s | perdida {p_tr:.3f} / val {p_va:.3f} | "
              f"madurez acc {acc:.3f} F1 {f1:.3f}", flush=True)
        if f1 > mejor:
            mejor, sin_mejora = f1, 0
            torch.save({"modelo": modelo.state_dict(), "epoca": ep, "config": vars(args), "val_acc": acc, "val_f1": f1},
                       os.path.join(salida, "best.pt"))
            print("   -> nuevo mejor modelo")
        else:
            sin_mejora += 1
            if sin_mejora >= args.patience:
                print(f"Parada temprana: {args.patience} epocas sin mejora.")
                break
    print(f"\nListo en {(time.time() - t_ini) / 60:.0f} min. Mejor F1 macro de validacion: {mejor:.4f}")
    print(f"Pesos: {os.path.join(salida, 'best.pt')}")


@torch.no_grad()
def evaluar(args, device):
    ck = torch.load(args.weights, map_location=device, weights_only=False)
    lado = ck["config"]["imgsz"]
    modelo = crear_modelo(False).to(device).eval()
    modelo.load_state_dict(ck["modelo"])
    n_param = sum(q.numel() for q in modelo.parameters())
    rutas = leer_lista(args.data_eval, args.split)
    if args.max_imagenes:
        rutas = rutas[:args.max_imagenes]
    pred = {}
    for n, ruta in enumerate(rutas):
        img, _ = letterbox(cv2.cvtColor(cv2.imread(ruta), cv2.COLOR_BGR2RGB), lado)
        x = normalizar(img)[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        prob = torch.softmax(modelo(x).float(), 1)[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        pred[os.path.basename(ruta)] = (int(prob.argmax()) + 1, float(prob.max()), ms, madurez_desde_archivo(ruta) + 1)
    reales = np.array([v[3] for v in pred.values()])
    preds = np.array([v[0] for v in pred.values()])
    lat = np.array([v[2] for v in pred.values()][3:] or [v[2] for v in pred.values()])
    acc = float((reales == preds).mean())
    f1 = float(f1_score(reales, preds, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0))
    print(f"Clasificador ResNet-34 ({n_param / 1e6:.2f} M parametros, entrada {lado} px) en {args.split}, n = {len(pred)}")
    print(f"Madurez: accuracy {acc:.4f} | F1 macro {f1:.4f} | +-1 nivel {float((np.abs(reales - preds) <= 1).mean()):.4f}")
    print(f"Latencia del clasificador: {lat.mean():.1f} ms/imagen en {device}")
    if not args.eval_segmentador:
        return
    # combinar con la evaluacion del segmentador (misma lista de imagenes)
    with open(os.path.join(args.eval_segmentador, "predicciones.csv"), encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    faltan = [r["archivo"] for r in filas if r["archivo"] not in pred]
    if faltan:
        sys.exit(f"{len(faltan)} imagenes del segmentador no estan en {args.split} (ej. {faltan[0]}).")
    lat_seg = np.array([float(r["latencia_ms"]) for r in filas])
    for r in filas:
        m, conf, ms, _ = pred[r["archivo"]]
        r["madurez_pred"], r["confianza_madurez"] = m, round(conf, 4)
        r["latencia_ms"] = round(float(r["latencia_ms"]) + ms, 2)
    out = os.path.abspath(args.out_dir or os.path.join(args.project, "dos_redes"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(args.eval_segmentador, "metricas.json"), encoding="utf-8") as f:
        seg = json.load(f)
    resumen = {
        "modelo": "YOLOv8s-seg (fallas) + ResNet-34 (madurez), dos redes", "segmentador": seg.get("weights"),
        "clasificador": args.weights, "parametros_clasificador_M": round(n_param / 1e6, 2), "split": args.split,
        "n_imagenes": len(filas), "pixel_con_roi": seg.get("pixel_con_roi"), "ocde": seg.get("ocde"),
        "ultralytics": seg.get("ultralytics"),
        "madurez": {"accuracy": round(acc, 4), "f1_macro": round(f1, 4),
                    "matriz_confusion": confusion_matrix(reales, preds, labels=[1, 2, 3, 4, 5]).tolist()},
        "latencia_ms": {"segmentador": round(float(lat_seg.mean()), 2), "clasificador": round(float(lat.mean()), 2),
                        "total": round(float(lat_seg.mean() + lat.mean()), 2), "dispositivo_clasificador": str(device),
                        "dispositivo_segmentador": (seg.get("latencia_ms") or {}).get("dispositivo")},
    }
    with open(os.path.join(out, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(f"Latencia total de las dos redes: {resumen['latencia_ms']['total']} ms/imagen "
          f"(segmentador {resumen['latencia_ms']['segmentador']} + clasificador {resumen['latencia_ms']['clasificador']})")
    print(f"Combinado guardado en: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join("data", "yolo_ampliado"), help="entrenamiento")
    p.add_argument("--data-eval", default=os.path.join("data", "yolo"), help="validacion y prueba (anotacion humana)")
    p.add_argument("--project", default=os.path.join("resultados", "comparacion"))
    p.add_argument("--name", default="resnet34_madurez")
    p.add_argument("--imgsz", type=int, default=448)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-batches", type=int, default=0)
    p.add_argument("--sin-preentrenado", action="store_true")
    p.add_argument("--evaluar", action="store_true")
    p.add_argument("--weights", default="")
    p.add_argument("--split", default="test")
    p.add_argument("--eval-segmentador", default="", help="carpeta eval_test_... del YOLOv8s-seg sin madurez")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-imagenes", type=int, default=0)
    p.add_argument("--device", default="")
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Dispositivo: {device}")
    if args.evaluar:
        if not args.weights:
            sys.exit("Falta --weights")
        evaluar(args, device)
    else:
        entrenar(args, device)


if __name__ == "__main__":
    main()
