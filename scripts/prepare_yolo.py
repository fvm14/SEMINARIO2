"""
prepare_yolo.py
Arma el dataset en el formato que espera Ultralytics a partir de data/raw y
del manifest (split por fruto), sin Roboflow.

Genera (por defecto en data/yolo/):
    images/*.jpg          imagenes (hardlink si se puede, si no copia)
    labels/*.txt          poligonos de labels_seg (clase 0 palta, clase 1 defecto)
    train.txt val.txt test.txt   rutas absolutas de las imagenes de cada split
    data.yaml

Ultralytics busca la etiqueta de cada imagen reemplazando /images/ por
/labels/ en la ruta, por eso se necesita esta carpeta aparte: data/raw/labels
tiene las cajas (bbox) originales, no los poligonos.

Uso:
    python scripts/prepare_yolo.py
"""

import argparse
import os
import shutil

import pandas as pd


def enlazar_o_copiar(origen, destino):
    if os.path.exists(destino):
        return
    try:
        os.link(origen, destino)
    except OSError:
        shutil.copy2(origen, destino)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join("data", "splits", "split_manifest.csv"))
    p.add_argument("--raw-dir", default=os.path.join("data", "raw"))
    p.add_argument("--dest", default=os.path.join("data", "yolo"))
    args = p.parse_args()

    dest = os.path.abspath(args.dest)
    img_dir, lbl_dir = os.path.join(dest, "images"), os.path.join(dest, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    ruta_excl = os.path.join(os.path.dirname(args.manifest), "anotaciones_excluidas.csv")
    if os.path.exists(ruta_excl):
        excluidas = set(pd.read_csv(ruta_excl).archivo)
        manifest = manifest[~manifest.archivo.isin(excluidas)]
        print(f"Excluyendo {len(excluidas)} imagenes con anotaciones rotas (ver {ruta_excl})")
    listas = {"train": [], "val": [], "test": []}
    for i, fila in enumerate(manifest.itertuples(index=False)):
        stem = fila.archivo.rsplit(".", 1)[0]
        img_dst = os.path.join(img_dir, fila.archivo)
        enlazar_o_copiar(os.path.join(args.raw_dir, "images", fila.archivo), img_dst)
        enlazar_o_copiar(os.path.join(args.raw_dir, "labels_seg", stem + ".txt"),
                         os.path.join(lbl_dir, stem + ".txt"))
        listas[fila.split].append(img_dst)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(manifest)}")

    for split, rutas in listas.items():
        with open(os.path.join(dest, f"{split}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(rutas)) + "\n")
        print(f"{split}: {len(rutas)} imagenes")

    yaml_path = os.path.join(dest, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {dest.replace(os.sep, '/')}\n"
                "train: train.txt\nval: val.txt\ntest: test.txt\n"
                "names:\n  0: palta\n  1: defecto\n")
    print(f"Listo: {yaml_path}")


if __name__ == "__main__":
    main()
