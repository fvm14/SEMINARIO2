"""
train.py
Entrenamiento del modelo multitarea (segmentacion palta/defecto + madurez 1-5).

Uso (desde la raiz del proyecto palta-multitarea):
    python scripts/train.py
    python scripts/train.py --epochs 100 --batch 8 --img-size 512

Prueba rapida (verifica que todo corre, 2 epocas cortas):
    python scripts/train.py --epochs 2 --max-batches 5 --workers 0

Salida en resultados/runs/<fecha_hora>/:
    best.pt       mejor checkpoint segun score de validacion
    last.pt       ultimo checkpoint (sirve para reanudar con --resume)
    historial.csv perdidas y metricas por epoca
    config.json   hiperparametros usados
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # modulos compartidos en scripts/


import argparse
import csv
import json
import math
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PaltaMultitaskDataset
from transforms import get_train_transforms, get_val_transforms
from model import PaltaMultitaskModel
from losses import PerdidaMultitarea
from metrics import AcumuladorMetricas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join("data", "splits", "split_manifest.csv"))
    p.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    p.add_argument("--out-dir", default=os.path.join("resultados", "runs"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4, help="lr de los cabezales")
    p.add_argument("--lr-encoder", type=float, default=1e-4, help="lr del backbone preentrenado")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--peso-madurez", type=float, default=1.0)
    p.add_argument("--peso-dice", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=0, help="solo para pruebas; 0 = todos")
    p.add_argument("--no-amp", action="store_true", help="desactiva mixed precision")
    p.add_argument("--resume", default="", help="ruta a last.pt para reanudar")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def crear_scheduler(optimizer, warmup_epochs, total_epochs, pasos_por_epoca):
    warmup = warmup_epochs * pasos_por_epoca
    total = total_epochs * pasos_por_epoca

    def factor(paso):
        if paso < warmup:
            return (paso + 1) / max(1, warmup)
        progreso = (paso - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progreso)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def ejecutar_epoca(modelo, loader, criterio, device, usar_amp, optimizer=None,
                   scaler=None, scheduler=None, max_batches=0, desc=""):
    entrenando = optimizer is not None
    modelo.train(entrenando)
    acumulador = AcumuladorMetricas()
    sumas = {"total": 0.0, "bce": 0.0, "dice": 0.0, "madurez": 0.0}
    n = 0

    with torch.set_grad_enabled(entrenando):
        for i, lote in enumerate(tqdm(loader, desc=desc, leave=False)):
            if max_batches and i >= max_batches:
                break
            imagenes = lote["imagen"].to(device, non_blocking=True)
            mascaras = lote["mascaras"].to(device, non_blocking=True)
            madurez = lote["madurez"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=usar_amp):
                salida = modelo(imagenes)
            # perdida en float32 para estabilidad numerica
            salida_f32 = {k: v.float() for k, v in salida.items()}
            perdidas = criterio(salida_f32, mascaras, madurez)

            if entrenando:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(perdidas["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            bs = imagenes.shape[0]
            for k in sumas:
                sumas[k] += perdidas[k].item() * bs
            n += bs
            acumulador.actualizar(salida_f32, mascaras, madurez)

    resultado = {f"loss_{k}": v / max(1, n) for k, v in sumas.items()}
    resultado.update(acumulador.calcular())
    return resultado


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usar_amp = device.type == "cuda" and not args.no_amp
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu":
        print("AVISO: no se detecto GPU CUDA. El entrenamiento sera MUY lento. "
              "Revisa que instalaste torch con soporte CUDA (ver requirements.txt).")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    ds_train = PaltaMultitaskDataset(args.manifest, args.raw_dir, "train", get_train_transforms(args.img_size))
    ds_val = PaltaMultitaskDataset(args.manifest, args.raw_dir, "val", get_val_transforms(args.img_size))
    print(f"Train: {len(ds_train)} | Val: {len(ds_val)}")

    kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True, drop_last=True, **kw)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, **kw)

    modelo = PaltaMultitaskModel(pretrained=True).to(device)
    criterio = PerdidaMultitarea(peso_dice=args.peso_dice, peso_madurez=args.peso_madurez)

    optimizer = torch.optim.AdamW([
        {"params": modelo.encoder.parameters(), "lr": args.lr_encoder},
        {"params": list(modelo.decoder_seg.parameters()) + list(modelo.cabezal_madurez.parameters()),
         "lr": args.lr},
    ], weight_decay=args.weight_decay)

    pasos_por_epoca = len(dl_train) if not args.max_batches else min(len(dl_train), args.max_batches)
    scheduler = crear_scheduler(optimizer, args.warmup_epochs, args.epochs, pasos_por_epoca)
    scaler = torch.amp.GradScaler(device.type, enabled=usar_amp)

    epoca_inicio, mejor_score, epocas_sin_mejora = 0, -1.0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        modelo.load_state_dict(ckpt["modelo"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        epoca_inicio = ckpt["epoca"] + 1
        mejor_score = ckpt["mejor_score"]
        epocas_sin_mejora = ckpt.get("epocas_sin_mejora", 0)
        run_dir = os.path.dirname(args.resume)
        print(f"Reanudando desde epoca {epoca_inicio} (mejor score {mejor_score:.4f})")
    else:
        run_dir = os.path.join(args.out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    ruta_hist = os.path.join(run_dir, "historial.csv")
    print(f"Resultados en: {run_dir}")

    for epoca in range(epoca_inicio, args.epochs):
        t0 = time.time()
        m_train = ejecutar_epoca(modelo, dl_train, criterio, device, usar_amp, optimizer,
                                 scaler, scheduler, args.max_batches, f"train {epoca+1}")
        m_val = ejecutar_epoca(modelo, dl_val, criterio, device, usar_amp,
                               max_batches=args.max_batches, desc=f"val {epoca+1}")
        dur = time.time() - t0

        fila = {"epoca": epoca + 1, "seg": round(dur, 1),
                "lr_encoder": optimizer.param_groups[0]["lr"], "lr_cabezales": optimizer.param_groups[1]["lr"]}
        fila.update({f"train_{k}": round(v, 5) for k, v in m_train.items()})
        fila.update({f"val_{k}": round(v, 5) for k, v in m_val.items()})

        nuevo = not os.path.exists(ruta_hist)
        with open(ruta_hist, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fila.keys()))
            if nuevo:
                w.writeheader()
            w.writerow(fila)

        print(f"[{epoca+1}/{args.epochs}] {dur:.0f}s | "
              f"loss train {m_train['loss_total']:.3f} val {m_val['loss_total']:.3f} | "
              f"IoU palta {m_val['iou_palta']:.3f} defecto {m_val['iou_defecto']:.3f} | "
              f"madurez acc {m_val['acc_madurez']:.3f} F1 {m_val['f1_madurez']:.3f} | "
              f"score {m_val['score']:.4f}")

        if m_val["score"] > mejor_score:
            mejor_score, epocas_sin_mejora = m_val["score"], 0
            torch.save({"modelo": modelo.state_dict(), "epoca": epoca, "metricas_val": m_val,
                        "config": vars(args)}, os.path.join(run_dir, "best.pt"))
            print(f"   -> nuevo mejor modelo guardado (score {mejor_score:.4f})")
        else:
            epocas_sin_mejora += 1

        torch.save({"modelo": modelo.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoca": epoca, "mejor_score": mejor_score,
                    "epocas_sin_mejora": epocas_sin_mejora, "config": vars(args)},
                   os.path.join(run_dir, "last.pt"))

        if epocas_sin_mejora >= args.patience:
            print(f"Early stopping: {args.patience} epocas sin mejora.")
            break

    print(f"Entrenamiento terminado. Mejor score val: {mejor_score:.4f}. Checkpoints en {run_dir}")


if __name__ == "__main__":
    main()
