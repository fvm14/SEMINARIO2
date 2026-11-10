"""
dos_caras_preparar.py
Evaluacion por fruto con ambas caras (paso 1: preparar la anotacion).

En el dataset de Xavier et al. (2024) cada palta se fotografio por los dos
lados el mismo dia (..._a_... y ..._b_...). De las 108 imagenes de prueba con
anotacion humana, solo 4 pares fruto-dia tienen anotados los dos lados; en los
otros 100 falta el lado opuesto. Este script:

  1. arma la lista de pares fruto-dia del conjunto de prueba (data/dos_caras/pares.csv);
  2. copia las 100 imagenes del lado faltante a data/dos_caras/por_anotar/images;
  3. las pre-anota SOLO con el contorno de la palta (modelo TFLite de rango
     dinamico). Los defectos NO se pre-anotan: se dibujan a mano para que la
     referencia no dependa del modelo que se evalua;
  4. deja data/dos_caras/por_anotar.zip listo para subir a Roboflow (YOLOv8).

Uso (desde la raiz del proyecto):
    python scripts/dos_caras_preparar.py
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia_movil import cargar_backend, letterbox, postprocesar  # noqa: E402
from ocde import fruto_completo  # noqa: E402

DIR_CRUDO = os.path.join("data", "raw", "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")


def original_de(nombre_roboflow):
    """T10_d02_275_a_1_jpg.rf.XXXX.jpg -> T10_d02_275_a_1.jpg"""
    return os.path.basename(nombre_roboflow).split("_jpg")[0] + ".jpg"


def otro_lado(original):
    p = original.split("_")
    p[3] = {"a": "b", "b": "a"}[p[3]]
    return "_".join(p)


def armar_pares(test_txt):
    with open(test_txt, encoding="utf-8") as f:
        anotadas = [os.path.basename(l.strip().replace("\\", "/")) for l in f if l.strip()]
    por_original = {original_de(a): os.path.splitext(a)[0] for a in anotadas}
    filas, vistos = [], set()
    for orig in sorted(por_original):
        p = orig[:-4].split("_")
        clave = (p[0], p[1], p[2])  # grupo, dia, fruto
        if clave in vistos:
            continue
        vistos.add(clave)
        a = "_".join([p[0], p[1], p[2], "a", p[4]]) + ".jpg"
        b = "_".join([p[0], p[1], p[2], "b", p[4]]) + ".jpg"
        filas.append({"fruto": f"{p[0]}_{p[2]}", "dia": p[1], "madurez": int(p[4]),
                      "lado_a": a, "lado_b": b,
                      "etiqueta_a": por_original.get(a, ""), "etiqueta_b": por_original.get(b, "")})
    return pd.DataFrame(filas)


def poligono_fruto(mascara):
    cnts, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=os.path.join("modelos_movil", "palta_multitarea_800_dynamic_range_quant.tflite"))
    ap.add_argument("--test", default=os.path.join("data", "yolo", "test.txt"))
    ap.add_argument("--out", default=os.path.join("data", "dos_caras"))
    ap.add_argument("--hilos", type=int, default=4)
    args = ap.parse_args()

    pares = armar_pares(args.test)
    os.makedirs(args.out, exist_ok=True)
    pares.to_csv(os.path.join(args.out, "pares.csv"), index=False)
    faltan = [r.lado_b if r.etiqueta_a else r.lado_a for r in pares.itertuples()
              if not (r.etiqueta_a and r.etiqueta_b)]
    print(f"Pares fruto-dia: {len(pares)} de {pares.fruto.nunique()} frutos | "
          f"con ambos lados anotados: {len(pares) - len(faltan)} | lados por anotar: {len(faltan)}")

    destino = os.path.join(args.out, "por_anotar")
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(destino, sub), exist_ok=True)
    backend = cargar_backend(args.modelo, args.hilos)
    for n, nombre in enumerate(faltan, 1):
        if os.path.exists(os.path.join(destino, "labels", nombre[:-4] + ".txt")):
            continue  # ya pre-anotada (permite reanudar)
        img = cv2.imread(os.path.join(DIR_CRUDO, nombre))
        shutil.copy2(os.path.join(DIR_CRUDO, nombre), os.path.join(destino, "images", nombre))
        x, r, pad = letterbox(img, backend.imgsz)
        det, proto, _ = backend(x)
        palta, defecto = postprocesar(det, proto, img.shape[:2], r, pad, 0.35, 0.99)
        fruto = fruto_completo(palta, defecto)
        poli = poligono_fruto(fruto)
        alto, ancho = img.shape[:2]
        with open(os.path.join(destino, "labels", nombre[:-4] + ".txt"), "w") as f:
            if poli is not None:
                f.write("0 " + " ".join(f"{px / ancho:.6f} {py / alto:.6f}" for px, py in poli) + "\n")
        if n % 20 == 0:
            print(f"  {n}/{len(faltan)}")

    with open(os.path.join(destino, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("train: images\nval: images\nnc: 2\nnames: ['palta', 'defecto']\n")
    zip_base = os.path.join(args.out, "por_anotar")
    shutil.make_archive(zip_base, "zip", destino)
    print(f"Listo: {zip_base}.zip ({len(faltan)} imagenes con el contorno de la palta pre-anotado)")


if __name__ == "__main__":
    main()
