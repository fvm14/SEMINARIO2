"""
auditar_anotaciones.py
Detecta anotaciones de labels_seg con errores evidentes (fallos de SAM2 al
convertir bbox -> poligono en Seminario I) y genera la lista de exclusion
data/splits/anotaciones_excluidas.csv, que usa prepare_yolo.py.

Criterios (imposibles en una anotacion correcta de una sola palta):
  - poligono de palta < 2.5% del area de la imagen (la mediana es ~16%): SAM2
    devolvio una mancha en vez del fruto;
  - area de defecto > area de palta: tipicamente el rectangulo de respaldo que
    AnotacionPolygon_Colab.py usaba cuando SAM2 fallaba, cubriendo fondo;
  - poligono de defecto de 4 vertices que llena su caja (>97%): el mismo
    rectangulo de respaldo, pero dentro de la palta. El modelo aprende a
    dibujar defectos rectangulares si se dejan.

No modifica el split: las imagenes excluidas simplemente no se usan, asi test
sigue siendo comparable entre modelos.

Uso:
    python scripts/auditar_anotaciones.py
"""

import os
import sys

import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rasterize_utils import poligonos_a_mascaras, leer_poligonos  # noqa: E402
import numpy as np  # noqa: E402

MIN_FRAC_PALTA = 0.025


def main():
    manifest = pd.read_csv(os.path.join("data", "splits", "split_manifest.csv"))
    filas = []
    for a, sp in zip(manifest.archivo, manifest.split):
        stem = a.rsplit(".", 1)[0]
        img = cv2.imread(os.path.join("data", "raw", "images", a))
        alto, ancho = img.shape[:2]
        gp, gd = poligonos_a_mascaras(os.path.join("data", "raw", "labels_seg", stem + ".txt"), alto, ancho)
        frac_palta = gp.sum() / (alto * ancho)
        ratio = gd.sum() / max(1, gp.sum())
        motivos = []
        if frac_palta < MIN_FRAC_PALTA:
            motivos.append(f"palta diminuta ({frac_palta*100:.2f}% de la imagen)")
        if ratio > 1:
            motivos.append(f"defecto mayor que la palta (ratio {ratio:.2f})")
        for clase, pts in leer_poligonos(os.path.join("data", "raw", "labels_seg", stem + ".txt")):
            if clase != 1:
                continue
            px = (pts * np.array([ancho, alto])).astype(np.float32)
            x, y, w, h = cv2.boundingRect(px)
            if len(px) <= 6 or (w * h and cv2.contourArea(px) / (w * h) > 0.97):
                motivos.append("defecto rectangular (respaldo de SAM2)")
                break
        if motivos:
            filas.append({"archivo": a, "split": sp, "motivo": "; ".join(motivos)})

    salida = os.path.join("data", "splits", "anotaciones_excluidas.csv")
    pd.DataFrame(filas, columns=["archivo", "split", "motivo"]).to_csv(salida, index=False)
    d = pd.DataFrame(filas)
    print(f"Excluidas: {len(d)} de {len(manifest)} | por split: {d.split.value_counts().to_dict() if len(d) else {}}")
    print(f"Lista guardada en {salida}")


if __name__ == "__main__":
    main()
