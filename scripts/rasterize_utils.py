"""
rasterize_utils.py
Convierte los poligonos de labels_seg/*.txt (formato YOLO-seg: clase x1 y1 x2 y2 ...,
coordenadas normalizadas 0-1) en mascaras binarias por clase.

Clase 0 = palta (fruto completo)
Clase 1 = defecto (puede haber varios poligonos de clase 1 por imagen; el
          defecto vive dentro del area de la palta, por eso las dos mascaras
          NO son mutuamente excluyentes)
"""

import numpy as np
import cv2


def leer_poligonos(ruta_txt: str):
    """Devuelve lista de (clase:int, puntos:np.ndarray[N,2] normalizados 0-1)."""
    poligonos = []
    with open(ruta_txt, "r") as f:
        for linea in f:
            datos = linea.strip().split()
            if not datos:
                continue
            clase = int(datos[0])
            coords = np.array(datos[1:], dtype=np.float32)
            if len(coords) < 6 or len(coords) % 2 != 0:
                # menos de 3 puntos o cantidad impar de valores: linea corrupta, se ignora
                continue
            puntos = coords.reshape(-1, 2)
            poligonos.append((clase, puntos))
    return poligonos


def poligonos_a_mascaras(ruta_txt: str, alto: int, ancho: int):
    """
    Rasteriza los poligonos de un archivo labels_seg a dos mascaras binarias
    (palta, defecto) del tamano (alto, ancho) dado, en escala [0, 1] float32
    listas para convertir a tensor.
    """
    mascara_palta = np.zeros((alto, ancho), dtype=np.uint8)
    mascara_defecto = np.zeros((alto, ancho), dtype=np.uint8)

    for clase, puntos in leer_poligonos(ruta_txt):
        puntos_px = puntos.copy()
        puntos_px[:, 0] *= ancho
        puntos_px[:, 1] *= alto
        puntos_px = puntos_px.astype(np.int32)

        destino = mascara_palta if clase == 0 else mascara_defecto
        cv2.fillPoly(destino, [puntos_px], color=1)

    return mascara_palta, mascara_defecto
