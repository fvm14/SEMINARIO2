"""
ocde.py
Logica OCDE y post-procesamiento ROI, portados de motor_ocde.py y
evaluar_con_roi.py (Seminario I) para aplicarlos sobre las mascaras que
PREDICE el modelo multitarea.

Umbrales OCDE (area estandar palta Hass ~42.4 cm2):
  Categoria I : defecto hasta 4 cm2 -> ratio <= 4/42.4 = 0.094
  Categoria II: defecto hasta 6 cm2 -> ratio <= 6/42.4 = 0.141
  Rechazado   : ratio > 0.141
"""

import numpy as np
import cv2

LIMITE_CAT_I = 0.094
LIMITE_CAT_II = 0.141
CATEGORIAS_OCDE = ["Categoria I", "Categoria II", "Rechazado"]
KERNEL_DILATACION_ROI = 20  # px, igual que evaluar_con_roi.py


def calcular_ratio(mascara_palta: np.ndarray, mascara_defecto: np.ndarray) -> float:
    """Area defecto / area palta en pixeles (igual que motor_ocde.calcular_ratio)."""
    px_palta = int(np.count_nonzero(mascara_palta))
    px_defecto = int(np.count_nonzero(mascara_defecto))
    if px_palta == 0:
        return 0.0
    return px_defecto / px_palta


def clasificar_ocde(ratio: float) -> int:
    """Devuelve el indice en CATEGORIAS_OCDE."""
    if ratio <= LIMITE_CAT_I:
        return 0
    if ratio <= LIMITE_CAT_II:
        return 1
    return 2


def filtrar_defecto_por_roi(mascara_palta: np.ndarray, mascara_defecto: np.ndarray,
                            kernel: int = KERNEL_DILATACION_ROI) -> np.ndarray:
    """
    Elimina pixeles de defecto que caen fuera de la palta predicha (dilatada
    `kernel` px para no recortar defectos en el borde del fruto). En Seminario I
    este truco redujo falsos positivos de defecto sobre el fondo.
    kernel=0 desactiva la dilatacion (recorte estricto al contorno).
    """
    palta = mascara_palta.astype(np.uint8)
    if kernel > 0:
        palta = cv2.dilate(palta, np.ones((kernel, kernel), np.uint8), iterations=1)
    return (mascara_defecto.astype(bool) & palta.astype(bool))


def mayor_componente(mascara: np.ndarray) -> np.ndarray:
    """Conserva solo la region conectada mas grande (hay una sola palta por
    imagen, asi que cualquier otra mancha predicha como palta es ruido)."""
    m = mascara.astype(np.uint8)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 2:  # fondo + 0 o 1 componente
        return m.astype(bool)
    mayor = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return etiquetas == mayor


def fruto_completo(mascara_palta: np.ndarray, mascara_defecto: np.ndarray) -> np.ndarray:
    """
    Region completa del fruto a partir de las mascaras predichas.

    El modelo se entrena con overlap_mask=True (default de Ultralytics): donde
    palta y defecto se superponen, el pixel queda para el defecto. Por eso la
    mascara de palta predicha tiene huecos donde hay defectos, mientras que el
    poligono anotado de palta cubre el fruto entero. Se reconstruye el fruto
    como la mayor region conectada de (palta U defecto), con huecos rellenados.
    """
    union = (mascara_palta.astype(bool) | mascara_defecto.astype(bool))
    fruto = mayor_componente(union).astype(np.uint8)
    contornos, _ = cv2.findContours(fruto, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lleno = np.zeros_like(fruto)
    cv2.drawContours(lleno, contornos, -1, 1, thickness=cv2.FILLED)
    return lleno.astype(bool)
