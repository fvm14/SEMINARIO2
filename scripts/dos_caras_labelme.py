"""
dos_caras_labelme.py
Puente entre las etiquetas YOLO-seg y LabelMe para anotar el lado faltante.

  --preparar : crea un .json de LabelMe junto a cada imagen de
               data/dos_caras/por_anotar/images con el contorno de la palta
               pre-anotado (clase "palta"). Los defectos se dibujan en LabelMe.
  --importar : convierte los .json anotados a YOLO-seg (0 palta, 1 defecto) en
               data/dos_caras/labels_seg/, que es lo que lee dos_caras_evaluar.py.
               Con --anotador <json> importa en cambio el archivo exportado por
               data/dos_caras/anotador.html (ver dos_caras_anotador.py).

Uso:
    python scripts/dos_caras_labelme.py --preparar
    labelme data/dos_caras/por_anotar/images --labels palta,defecto --nodata --autosave
    python scripts/dos_caras_labelme.py --importar
"""

import argparse
import glob
import json
import os

import cv2

DIR = os.path.join("data", "dos_caras")
IMG = os.path.join(DIR, "por_anotar", "images")
LBL = os.path.join(DIR, "por_anotar", "labels")
SALIDA = os.path.join(DIR, "labels_seg")
CLASES = {"palta": 0, "defecto": 1}


def preparar():
    n = 0
    for ruta_img in sorted(glob.glob(os.path.join(IMG, "*.jpg"))):
        ruta_json = ruta_img[:-4] + ".json"
        if os.path.exists(ruta_json):
            continue  # no pisar lo ya anotado
        alto, ancho = cv2.imread(ruta_img).shape[:2]
        formas = []
        ruta_txt = os.path.join(LBL, os.path.basename(ruta_img)[:-4] + ".txt")
        if os.path.exists(ruta_txt):
            for linea in open(ruta_txt):
                v = linea.split()
                if len(v) >= 7 and v[0] == "0":
                    xy = [float(t) for t in v[1:]]
                    pts = [[xy[i] * ancho, xy[i + 1] * alto] for i in range(0, len(xy), 2)]
                    formas.append({"label": "palta", "points": pts, "group_id": None,
                                   "description": "", "shape_type": "polygon", "flags": {}, "mask": None})
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump({"version": "5.5.0", "flags": {}, "shapes": formas,
                       "imagePath": os.path.basename(ruta_img), "imageData": None,
                       "imageHeight": alto, "imageWidth": ancho}, f, indent=1)
        n += 1
    print(f"{n} archivos .json creados en {IMG}")


def importar_anotador(ruta_json, todas=False):
    """JSON exportado por anotador.html -> YOLO-seg."""
    os.makedirs(SALIDA, exist_ok=True)
    d = json.load(open(ruta_json, encoding="utf-8"))["imagenes"]
    pendientes, sin_palta, sanas = [], [], 0
    for nombre, a in sorted(d.items()):
        if not a.get("lista") and not todas:
            pendientes.append(nombre)
            continue
        ancho, alto = a["ancho"], a["alto"]
        lineas = []
        for c, polis in ((0, [a["palta"]] if a.get("palta") else []), (1, a.get("defectos", []))):
            for p in polis:
                if len(p) >= 3:
                    lineas.append(f"{c} " + " ".join(f"{x / ancho:.6f} {y / alto:.6f}" for x, y in p))
        if not a.get("palta"):
            sin_palta.append(nombre)
        sanas += not a.get("defectos")
        with open(os.path.join(SALIDA, nombre[:-4] + ".txt"), "w") as f:
            f.write("\n".join(lineas) + ("\n" if lineas else ""))
    print(f"{len(d) - len(pendientes)} imagenes importadas a {SALIDA} ({sanas} sin defecto)")
    sin_def = [n for n, a in sorted(d.items()) if not a.get("defectos") and n not in pendientes]
    if sin_def:
        print("  Sin defecto (confirma que sean frutos sanos y no imagenes sin revisar):")
        for n in sin_def:
            print("    " + n)
    if pendientes:
        print(f"  {len(pendientes)} sin marcar como listas (no se importaron): {pendientes[:5]}...")
    if sin_palta:
        print(f"  SIN contorno de palta: {sin_palta}")


def importar():
    os.makedirs(SALIDA, exist_ok=True)
    jsons = sorted(glob.glob(os.path.join(IMG, "*.json")))
    resumen = {"imagenes": 0, "sin_palta": [], "sin_defecto": 0, "etiqueta_desconocida": set()}
    for ruta in jsons:
        d = json.load(open(ruta, encoding="utf-8"))
        alto, ancho = d["imageHeight"], d["imageWidth"]
        lineas, n_palta, n_def = [], 0, 0
        for s in d["shapes"]:
            etiqueta = s["label"].strip().lower()
            if etiqueta not in CLASES:
                resumen["etiqueta_desconocida"].add(s["label"])
                continue
            if s.get("shape_type", "polygon") != "polygon" or len(s["points"]) < 3:
                continue
            c = CLASES[etiqueta]
            n_palta += c == 0
            n_def += c == 1
            lineas.append(f"{c} " + " ".join(f"{x / ancho:.6f} {y / alto:.6f}" for x, y in s["points"]))
        with open(os.path.join(SALIDA, os.path.basename(ruta)[:-5] + ".txt"), "w") as f:
            f.write("\n".join(lineas) + ("\n" if lineas else ""))
        resumen["imagenes"] += 1
        if not n_palta:
            resumen["sin_palta"].append(os.path.basename(ruta))
        resumen["sin_defecto"] += n_def == 0
    print(f"{resumen['imagenes']} imagenes importadas a {SALIDA}")
    print(f"  sin defecto anotado: {resumen['sin_defecto']} (revisa que sean frutos realmente sanos)")
    if resumen["sin_palta"]:
        print(f"  SIN contorno de palta: {resumen['sin_palta']}")
    if resumen["etiqueta_desconocida"]:
        print(f"  etiquetas ignoradas (usa 'palta' o 'defecto'): {sorted(resumen['etiqueta_desconocida'])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--preparar", action="store_true")
    g.add_argument("--importar", action="store_true")
    ap.add_argument("--anotador", default="", help="JSON exportado por anotador.html")
    ap.add_argument("--todas", action="store_true",
                    help="importar todas las imagenes del JSON, aunque no esten marcadas como listas")
    a = ap.parse_args()
    if a.preparar:
        preparar()
    elif a.anotador:
        importar_anotador(a.anotador, a.todas)
    else:
        importar()
