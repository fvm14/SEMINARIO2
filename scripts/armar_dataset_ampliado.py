"""
armar_dataset_ampliado.py
Junta las anotaciones humanas (data/yolo) con las pseudo-etiquetas aceptadas
(data/pseudo) en un dataset YOLO nuevo: data/yolo_ampliado/.

  train = anotaciones humanas de train + pseudo-etiquetas aceptadas
  val   = igual que data/yolo (solo anotacion humana)
  test  = igual que data/yolo (solo anotacion humana)

Aceptadas = estado "auto" segun la regla calibrada en validacion. Si existe
data/pseudo/revision_decisiones.csv (exportado desde revision.html), se aplica
encima: "aprobada" agrega la foto aunque fuera dudosa, "rechazada" y "corregir"
la sacan de esta ronda.

Uso (despues de generar_pseudoetiquetas.py):
    python scripts/armar_dataset_ampliado.py
"""

import argparse
import os
import shutil

import pandas as pd

DIR_CRUDO = os.path.join("data", "raw", "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")


def enlazar_o_copiar(origen, destino):
    if os.path.exists(destino):
        os.remove(destino)
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join("data", "yolo"))
    ap.add_argument("--pseudo", default=os.path.join("data", "pseudo"))
    ap.add_argument("--dest", default=os.path.join("data", "yolo_ampliado"))
    args = ap.parse_args()

    dest = os.path.abspath(args.dest)
    img_dir, lbl_dir = os.path.join(dest, "images"), os.path.join(dest, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    p = pd.read_csv(os.path.join(args.pseudo, "pseudoetiquetas.csv"))
    aceptadas = set(p[p.estado == "auto"].archivo)
    ruta_dec = os.path.join(args.pseudo, "revision_decisiones.csv")
    n_ap = n_fuera = 0
    if os.path.exists(ruta_dec):
        dec = pd.read_csv(ruta_dec)
        for a, dcs in zip(dec.archivo, dec.decision):
            if dcs == "aprobada":
                n_ap += a not in aceptadas
                aceptadas.add(a)
            elif a in aceptadas:
                aceptadas.discard(a)
                n_fuera += 1
        print(f"Revision manual aplicada: +{n_ap} aprobadas a mano, -{n_fuera} automaticas rechazadas/para corregir")
    else:
        print("Sin revision_decisiones.csv: solo se usan las aceptadas automaticamente")

    listas = {}
    for split in ("train", "val", "test"):
        with open(os.path.join(args.base, f"{split}.txt"), encoding="utf-8") as f:
            listas[split] = [l.strip() for l in f if l.strip()]
    base_img = os.path.dirname(listas["train"][0])
    base_lbl = base_img.replace(os.sep + "images", os.sep + "labels").replace("/images", "/labels")

    # anotaciones humanas (todos los splits) -> mismo nombre en el dataset ampliado
    for split in listas:
        nuevas = []
        for ruta in listas[split]:
            nombre = os.path.basename(ruta)
            stem = os.path.splitext(nombre)[0]
            enlazar_o_copiar(ruta, os.path.join(img_dir, nombre))
            enlazar_o_copiar(os.path.join(base_lbl, stem + ".txt"), os.path.join(lbl_dir, stem + ".txt"))
            nuevas.append(os.path.join(img_dir, nombre))
        listas[split] = nuevas
    n_humanas = len(listas["train"])

    for a in sorted(aceptadas):
        stem = os.path.splitext(a)[0]
        enlazar_o_copiar(os.path.join(DIR_CRUDO, a), os.path.join(img_dir, a))
        enlazar_o_copiar(os.path.join(args.pseudo, "labels", stem + ".txt"), os.path.join(lbl_dir, stem + ".txt"))
        listas["train"].append(os.path.join(img_dir, a))

    for split, rutas in listas.items():
        with open(os.path.join(dest, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(rutas) + "\n")
    with open(os.path.join(dest, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(f"path: {dest.replace(os.sep, '/')}\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
                "names:\n  0: palta\n  1: defecto\n")
    print(f"train: {len(listas['train'])} ({n_humanas} humanas + {len(aceptadas)} pseudo) | "
          f"val: {len(listas['val'])} | test: {len(listas['test'])}")
    print(f"Listo: {os.path.join(dest, 'data.yaml')}")


if __name__ == "__main__":
    main()
