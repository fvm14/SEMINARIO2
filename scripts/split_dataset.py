"""
split_dataset.py
Reemplaza el export/split de Roboflow. Genera un manifest CSV que asigna
cada imagen a train/val/test (~70/20/10).

El split se hace POR FRUTO, no por imagen: el dataset Xavier et al. fotografia
la misma palta varios dias y por ambos lados (ej. T10_d01_414_a, T10_d09_414_b,
T10_d20_414_a son la palta 414 del grupo T10). Si esas fotos quedan repartidas
entre train y test, el modelo se evalua con frutos que ya vio (fuga de datos).
Se usa StratifiedGroupKFold (10 folds: 7 train, 2 val, 1 test) para que ademas
la distribucion de madurez quede balanceada entre subconjuntos.

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
from collections import defaultdict, Counter

from sklearn.model_selection import StratifiedGroupKFold

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


def extraer_fruto(nombre_archivo: str) -> str:
    """T10_d04_414_b_1_jpg... -> 'T10_414' (grupo de almacenamiento + id del fruto)."""
    partes = nombre_archivo.split("_")
    return f"{partes[0]}_{partes[2]}"


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

    archivos, niveles, frutos = [], [], []
    for nivel, lista in grupos.items():
        for img in lista:
            archivos.append(img)
            niveles.append(nivel)
            frutos.append(extraer_fruto(img))
    print(f"Frutos unicos: {len(set(frutos))}")

    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
    fold_de = {}
    for k, (_, idx_fold) in enumerate(sgkf.split(archivos, niveles, groups=frutos)):
        for i in idx_fold:
            fold_de[i] = k
    # folds 0-6 -> train, 7-8 -> val, 9 -> test
    split_de_fold = {k: "train" for k in range(7)}
    split_de_fold.update({7: "val", 8: "val", 9: "test"})

    manifest_rows = []
    conteo_splits = Counter()
    conteo_nivel = defaultdict(Counter)
    for i, img in enumerate(archivos):
        sp = split_de_fold[fold_de[i]]
        manifest_rows.append({"archivo": img, "split": sp, "madurez": niveles[i], "fruto": frutos[i]})
        conteo_splits[sp] += 1
        conteo_nivel[niveles[i]][sp] += 1
    manifest_rows.sort(key=lambda r: (r["split"], r["madurez"], r["archivo"]))

    for nivel in sorted(conteo_nivel):
        c = conteo_nivel[nivel]
        print(f"Nivel {nivel}: train={c['train']} val={c['val']} test={c['test']}")

    frutos_por_split = defaultdict(set)
    for r in manifest_rows:
        frutos_por_split[r["split"]].add(r["fruto"])
    cruce = (frutos_por_split["train"] & frutos_por_split["val"]) | \
            (frutos_por_split["train"] & frutos_por_split["test"]) | \
            (frutos_por_split["val"] & frutos_por_split["test"])
    assert not cruce, f"Frutos repetidos entre splits: {sorted(cruce)[:5]}"
    print("Verificacion: ningun fruto aparece en mas de un split.")

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "split_manifest.csv")
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["archivo", "split", "madurez", "fruto"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n" + "=" * 50)
    print("RESUMEN DEL SPLIT (solo manifest, sin copiar archivos)")
    print("=" * 50)
    for split_name in ["train", "val", "test"]:
        print(f"{split_name}: {conteo_splits[split_name]} imagenes, {len(frutos_por_split[split_name])} frutos")
    print(f"\nManifest guardado en: {manifest_path}")


if __name__ == "__main__":
    main()
