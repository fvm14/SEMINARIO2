"""
split_dataset.py
Reemplaza el export/split de Roboflow. Genera un manifest CSV que asigna
cada imagen a train/val/test (70/20/10), estratificado por nivel de madurez.

NO copia archivos fisicamente (data/raw/ es la unica copia de imagenes y
labels_seg). El Dataset de PyTorch lee data/raw/ y filtra usando este
manifest, evitando duplicar ~2,600 archivos.

Uso:
    python scripts/split_dataset.py

Espera:
    data/raw/images/*.jpg
    data/raw/labels_seg/*.txt

Genera:
    data/splits/split_manifest.csv  (columnas: archivo, split, madurez)
"""

import os
import random
import csv
from collections import defaultdict

RANDOM_SEED = 42
RAW_DIR = os.path.join("data", "raw")
IMAGES_DIR = os.path.join(RAW_DIR, "images")
LABELS_SEG_DIR = os.path.join(RAW_DIR, "labels_seg")
OUT_DIR = os.path.join("data", "splits")

TRAIN_PCT = 0.70
VAL_PCT = 0.20
TEST_PCT = 0.10


def extraer_madurez(nombre_archivo: str):
    base = nombre_archivo.rsplit(".", 1)[0]
    partes = base.split("_")
    try:
        nivel = int(partes[4])
        if nivel in (1, 2, 3, 4, 5):
            return nivel
    except (IndexError, ValueError):
        pass
    return None


def tiene_clase_palta(ruta_txt: str) -> bool:
    with open(ruta_txt, "r") as f:
        for linea in f:
            datos = linea.strip().split()
            if datos and datos[0] == "0":
                return True
    return False


def main():
    random.seed(RANDOM_SEED)

    imagenes = sorted(f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg"))
    print(f"Imagenes encontradas: {len(imagenes)}")

    grupos = defaultdict(list)
    excluidas = []

    for img_name in imagenes:
        stem = img_name.rsplit(".", 1)[0]
        txt_name = stem + ".txt"
        txt_path = os.path.join(LABELS_SEG_DIR, txt_name)

        if not os.path.exists(txt_path):
            excluidas.append((img_name, "sin labels_seg"))
            continue

        nivel = extraer_madurez(img_name)
        if nivel is None:
            excluidas.append((img_name, "madurez no parseable"))
            continue

        if not tiene_clase_palta(txt_path):
            excluidas.append((img_name, "sin poligono de palta (clase 0)"))
            continue

        grupos[nivel].append(img_name)

    print(f"Excluidas: {len(excluidas)}")
    for img_name, motivo in excluidas:
        print(f"  - {img_name}: {motivo}")

    total_validas = sum(len(v) for v in grupos.values())
    print(f"Imagenes validas para split: {total_validas}")

    manifest_rows = []
    conteo_splits = {"train": 0, "val": 0, "test": 0}

    for nivel in sorted(grupos.keys()):
        lista = grupos[nivel][:]
        random.shuffle(lista)
        n = len(lista)
        n_train = round(n * TRAIN_PCT)
        n_val = round(n * VAL_PCT)
        n_test = n - n_train - n_val

        train_i = lista[:n_train]
        val_i = lista[n_train:n_train + n_val]
        test_i = lista[n_train + n_val:]

        print(f"Nivel {nivel}: total={n} -> train={len(train_i)} val={len(val_i)} test={len(test_i)}")

        for img in train_i:
            manifest_rows.append({"archivo": img, "split": "train", "madurez": nivel})
        for img in val_i:
            manifest_rows.append({"archivo": img, "split": "val", "madurez": nivel})
        for img in test_i:
            manifest_rows.append({"archivo": img, "split": "test", "madurez": nivel})

        conteo_splits["train"] += len(train_i)
        conteo_splits["val"] += len(val_i)
        conteo_splits["test"] += len(test_i)

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["archivo", "split", "madurez"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 50)
    print("RESUMEN DEL SPLIT (solo manifest, sin copiar archivos)")
    print("=" * 50)
    for split_name, n in conteo_splits.items():
        print(f"{split_name}: {n} imagenes")
    print(f"\nManifest guardado en: {manifest_path}")


if __name__ == "__main__":
    main()
