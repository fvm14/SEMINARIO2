"""
referencia_app.py
Resultado de referencia de inferencia_movil.analizar() para una o varias fotos,
para comparar contra lo que muestra la app PaltaScan con el mismo .tflite
(categoria OCDE, ratio, madurez y pixeles de fruto/defecto).

La app trabaja en la resolucion del letterbox y Python en la de la foto: el
ratio puede diferir en milesimas; la categoria y la madurez deben coincidir.

Uso:
    python scripts/referencia_app.py --modelo modelos_movil/palta_multitarea_800_dynamic_range_quant.tflite --fotos carpeta_o_foto.jpg --conf-defecto 0.05
"""

import argparse
import glob
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia_movil import analizar, cargar_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--fotos", required=True, help="una imagen o una carpeta")
    ap.add_argument("--conf-defecto", type=float, default=0.05)
    ap.add_argument("--salida", default="")
    args = ap.parse_args()

    rutas = sorted(glob.glob(os.path.join(args.fotos, "*.jpg")) + glob.glob(os.path.join(args.fotos, "*.png"))) \
        if os.path.isdir(args.fotos) else [args.fotos]
    backend = cargar_backend(args.modelo, 4)
    filas = []
    for ruta in rutas:
        r = analizar(backend, cv2.imread(ruta), 0.35, args.conf_defecto)
        filas.append({
            "foto": os.path.basename(ruta), "ocde": r["ocde"], "ratio": round(r["ratio"], 4),
            "madurez": r["madurez"], "prob_madurez": round(r["prob_madurez"], 4),
            "px_fruto": int(r["fruto"].sum()), "px_defecto": int(r["defecto"].sum()), "ms_inferencia": round(r["ms"], 1),
        })
        print(json.dumps(filas[-1], ensure_ascii=False))
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            json.dump(filas, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
