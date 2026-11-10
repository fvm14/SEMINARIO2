"""
dos_caras_evaluar.py
Evaluacion por fruto con ambas caras (paso 2).

Unidad de analisis: el par fruto-dia (lado a + lado b fotografiados el mismo
dia). Para cada par se compara:

  una cara  -> la categoria del fruto se decide con una sola foto, como hasta
               ahora (se evaluan ambos lados por separado y se promedia su acierto)
  dos caras -> el area de defecto se suma en las dos fotos y la madurez se
               decide con el promedio de las probabilidades de ambas

Regla de dos caras. Cada foto muestra casi la mitad del fruto y su ratio
defecto/fruto se normaliza por el area proyectada de referencia (42.4 cm2), de
modo que el defecto total es (ratio_a + ratio_b) x 42.4 cm2. Comparado con los
4 y 6 cm2 de la norma, equivale a:
  Categoria I  si ratio_a + ratio_b <= 0.094
  Categoria II si ratio_a + ratio_b <= 0.141
  Rechazado    en otro caso
La categoria REAL del fruto se calcula con la misma regla sobre la anotacion
humana de ambos lados.

Etiquetas humanas: data/raw/labels_seg (las 108 originales) y las del lado
faltante en data/dos_caras/labels_seg (anotadas en LabelMe e importadas con
dos_caras_labelme.py; tambien se acepta una exportacion YOLOv8 en data/dos_caras/roboflow).
Sin esa exportacion, el script solo reporta el analisis sin anotacion: en
cuantos pares cambia la categoria predicha al agregar el segundo lado.

Uso:
    python scripts/dos_caras_evaluar.py
    python scripts/dos_caras_evaluar.py --modelo modelos_movil/palta_multitarea_800_dynamic_range_quant.tflite --conf-defecto 0.05
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia_movil import cargar_backend, letterbox, postprocesar  # noqa: E402
from ocde import (CATEGORIAS_OCDE, calcular_ratio, clasificar_ocde,  # noqa: E402
                  filtrar_defecto_por_roi, fruto_completo)
from rasterize_utils import poligonos_a_mascaras  # noqa: E402

DIR_CRUDO = os.path.join("data", "raw", "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
CATS = CATEGORIAS_OCDE


def cat(ratio):
    return CATS[clasificar_ocde(ratio)]


def indice_etiquetas(labels_seg, dir_roboflow):
    """original.jpg -> ruta del .txt con anotacion humana."""
    idx = {}
    rutas = glob.glob(os.path.join(labels_seg, "*.txt"))
    rutas += glob.glob(os.path.join(dir_roboflow, "**", "labels", "*.txt"), recursive=True)
    for r in rutas:
        base = os.path.basename(r)
        orig = base.split("_jpg")[0] + ".jpg" if "_jpg" in base else base[:-4] + ".jpg"
        idx.setdefault(orig, r)
    return idx


def predecir(backend, nombre, conf_defecto):
    img = cv2.imread(os.path.join(DIR_CRUDO, nombre))
    x, r, pad = letterbox(img, backend.imgsz)
    det, proto, mad = backend(x)
    palta, defecto = postprocesar(det, proto, img.shape[:2], r, pad, 0.35, conf_defecto)
    fruto = fruto_completo(palta, defecto)
    defecto = filtrar_defecto_por_roi(fruto, defecto)
    return calcular_ratio(fruto, defecto), np.asarray(mad, float), img.shape[:2]


def metricas(d):
    """d: una fila por par, con columnas *_una (acierto promedio de los dos lados) y *_dos."""
    out = {}
    for modo in ("una", "dos"):
        rech = d.ocde_real_fruto == "Rechazado"
        if modo == "dos":
            ok = (d.ocde_pred_dos == d.ocde_real_fruto).astype(float)
            f1 = f1_score(d.ocde_real_fruto, d.ocde_pred_dos, labels=CATS, average="macro", zero_division=0)
            rr = (d.ocde_pred_dos[rech] == "Rechazado").mean() if rech.any() else np.nan
            mok = (d.madurez_pred_dos == d.madurez).astype(float)
            mf1 = f1_score(d.madurez, d.madurez_pred_dos, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)
        else:
            ok = ((d.ocde_pred_a == d.ocde_real_fruto).astype(float) + (d.ocde_pred_b == d.ocde_real_fruto)) / 2
            yr = pd.concat([d.ocde_real_fruto, d.ocde_real_fruto]); yp = pd.concat([d.ocde_pred_a, d.ocde_pred_b])
            f1 = f1_score(yr, yp, labels=CATS, average="macro", zero_division=0)
            rr = ((d.ocde_pred_a[rech] == "Rechazado").mean() + (d.ocde_pred_b[rech] == "Rechazado").mean()) / 2 \
                if rech.any() else np.nan
            mok = ((d.madurez_pred_a == d.madurez).astype(float) + (d.madurez_pred_b == d.madurez)) / 2
            mr = pd.concat([d.madurez, d.madurez]); mp = pd.concat([d.madurez_pred_a, d.madurez_pred_b])
            mf1 = f1_score(mr, mp, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)
        out[f"ocde_acierto_{modo}"] = float(ok.mean())
        out[f"ocde_f1_macro_{modo}"] = float(f1)
        out[f"recall_rechazado_{modo}"] = float(rr)
        out[f"madurez_exactitud_{modo}"] = float(mok.mean())
        out[f"madurez_f1_macro_{modo}"] = float(mf1)
    # limite de una sola cara con anotacion perfecta: cuanto se equivoca la categoria
    # del fruto si solo se mira un lado, aun sin error del modelo
    out["ocde_acierto_una_cara_real"] = float((((d.ocde_real_a == d.ocde_real_fruto).astype(float)
                                               + (d.ocde_real_b == d.ocde_real_fruto)) / 2).mean())
    out["mae_ratio_total"] = float((d.ratio_real_a + d.ratio_real_b - d.ratio_pred_a - d.ratio_pred_b).abs().mean())
    return out


def bootstrap(d, n, seed):
    rng = np.random.default_rng(seed)
    grupos = list(d.groupby("fruto").indices.values())
    filas = []
    for _ in range(n):
        idx = np.concatenate([grupos[i] for i in rng.integers(0, len(grupos), len(grupos))])
        filas.append(metricas(d.iloc[idx]))
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=os.path.join("modelos_movil", "palta_multitarea_800_dynamic_range_quant.tflite"))
    ap.add_argument("--conf-defecto", type=float, default=0.05)
    ap.add_argument("--pares", default=os.path.join("data", "dos_caras", "pares.csv"))
    ap.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    ap.add_argument("--roboflow", default=os.path.join("data", "dos_caras", "roboflow"))
    ap.add_argument("--labels-dos-caras", default=os.path.join("data", "dos_caras", "labels_seg"),
                    help="etiquetas del lado faltante importadas desde LabelMe")
    ap.add_argument("--out", default=os.path.join("resultados", "dos_caras"))
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    pares = pd.read_csv(args.pares)
    etiquetas = indice_etiquetas(args.labels_seg, args.roboflow)
    for r in glob.glob(os.path.join(args.labels_dos_caras, "*.txt")):
        etiquetas.setdefault(os.path.basename(r)[:-4] + ".jpg", r)

    # predicciones por foto (cache para no repetir la inferencia)
    nombre_modelo = os.path.splitext(os.path.basename(args.modelo))[0]
    ruta_cache = os.path.join(args.out, f"predicciones_por_foto_{nombre_modelo}_{args.conf_defecto}.csv")
    cache = pd.read_csv(ruta_cache).set_index("archivo") if os.path.exists(ruta_cache) else None
    backend = None
    filas_foto = []
    for nombre in sorted(set(pares.lado_a) | set(pares.lado_b)):
        if cache is not None and nombre in cache.index:
            c = cache.loc[nombre]
            filas_foto.append(c.to_dict() | {"archivo": nombre})
            continue
        backend = backend or cargar_backend(args.modelo, args.hilos)
        ratio, mad, _ = predecir(backend, nombre, args.conf_defecto)
        filas_foto.append({"archivo": nombre, "ratio_pred": ratio,
                           **{f"p{k + 1}": float(mad[k]) for k in range(5)}})
        if len(filas_foto) % 25 == 0:
            print(f"  {len(filas_foto)} fotos")
    fotos = pd.DataFrame(filas_foto).set_index("archivo")
    fotos.reset_index().to_csv(ruta_cache, index=False)

    def real(nombre):
        ruta = etiquetas.get(nombre)
        if ruta is None:
            return np.nan
        alto, ancho = cv2.imread(os.path.join(DIR_CRUDO, nombre)).shape[:2]
        gp, gd = poligonos_a_mascaras(ruta, alto, ancho)
        return calcular_ratio(gp.astype(bool), gd.astype(bool))

    filas = []
    for p in pares.itertuples():
        fa, fb = fotos.loc[p.lado_a], fotos.loc[p.lado_b]
        pa = fa[[f"p{k}" for k in range(1, 6)]].to_numpy(float)
        pb = fb[[f"p{k}" for k in range(1, 6)]].to_numpy(float)
        ra, rb = real(p.lado_a), real(p.lado_b)
        filas.append({
            "fruto": p.fruto, "dia": p.dia, "madurez": p.madurez, "lado_a": p.lado_a, "lado_b": p.lado_b,
            "ratio_pred_a": fa.ratio_pred, "ratio_pred_b": fb.ratio_pred,
            "ratio_real_a": ra, "ratio_real_b": rb,
            "ocde_pred_a": cat(fa.ratio_pred), "ocde_pred_b": cat(fb.ratio_pred),
            "ocde_pred_dos": cat(fa.ratio_pred + fb.ratio_pred),
            "ocde_real_a": cat(ra) if ra == ra else "", "ocde_real_b": cat(rb) if rb == rb else "",
            "ocde_real_fruto": cat(ra + rb) if (ra == ra and rb == rb) else "",
            "madurez_pred_a": int(pa.argmax()) + 1, "madurez_pred_b": int(pb.argmax()) + 1,
            "madurez_pred_dos": int((pa + pb).argmax()) + 1,
        })
    d = pd.DataFrame(filas)
    d.to_csv(os.path.join(args.out, "predicciones_por_fruto.csv"), index=False)

    # 1) sin anotacion: cuanto cambia la decision al agregar el segundo lado
    cambio_a = (d.ocde_pred_a != d.ocde_pred_dos).mean()
    cambio_b = (d.ocde_pred_b != d.ocde_pred_dos).mean()
    lados_distintos = (d.ocde_pred_a != d.ocde_pred_b).mean()
    res = {"modelo": args.modelo, "conf_defecto": args.conf_defecto, "n_pares": len(d),
           "n_frutos": int(d.fruto.nunique()),
           "sin_anotacion": {
               "pares_con_categoria_distinta_entre_lados": round(float(lados_distintos), 4),
               "cambio_al_agregar_lado_b": round(float(cambio_a), 4),
               "cambio_al_agregar_lado_a": round(float(cambio_b), 4),
               "distribucion_pred_una_cara": pd.concat([d.ocde_pred_a, d.ocde_pred_b]).value_counts().to_dict(),
               "distribucion_pred_dos_caras": d.ocde_pred_dos.value_counts().to_dict()}}
    print(f"\n{len(d)} pares fruto-dia de {d.fruto.nunique()} frutos")
    print(f"Categoria predicha distinta entre lado a y lado b: {lados_distintos:.1%}")
    print(f"Cambia la categoria al agregar el segundo lado: {(cambio_a + cambio_b) / 2:.1%} (promedio)")

    # 2) con anotacion humana de ambos lados
    completos = d[d.ocde_real_fruto != ""].reset_index(drop=True)
    res["n_pares_con_ambos_lados_anotados"] = len(completos)
    if len(completos) >= 20:
        puntual = metricas(completos)
        bs = bootstrap(completos, args.n_bootstrap, args.seed)
        res["con_anotacion"] = {k: {"valor": round(v, 4),
                                    "ic95": [round(float(np.nanpercentile(bs[k], 2.5)), 4),
                                             round(float(np.nanpercentile(bs[k], 97.5)), 4)]}
                                for k, v in puntual.items()}
        dif = {}
        for m in ("ocde_acierto", "ocde_f1_macro", "recall_rechazado", "madurez_exactitud", "madurez_f1_macro"):
            v = bs[f"{m}_dos"] - bs[f"{m}_una"]
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            dif[m] = {"dos_menos_una": round(puntual[f"{m}_dos"] - puntual[f"{m}_una"], 4),
                      "ic95": [round(float(lo), 4), round(float(hi), 4)], "significativa": bool(lo > 0 or hi < 0)}
        res["diferencia_dos_menos_una"] = dif
        res["distribucion_real"] = {
            "una_cara": pd.concat([completos.ocde_real_a, completos.ocde_real_b]).value_counts().to_dict(),
            "fruto_dos_caras": completos.ocde_real_fruto.value_counts().to_dict()}
        res["matriz_confusion_dos_caras"] = {
            "etiquetas": CATS,
            "matriz": confusion_matrix(completos.ocde_real_fruto, completos.ocde_pred_dos, labels=CATS).tolist()}
        print(f"\nCon anotacion de ambos lados: {len(completos)} pares")
        print(f"{'metrica':26s} {'una cara':>9s} {'dos caras':>10s}   dos - una [IC 95%]")
        for m, v in dif.items():
            print(f"{m:26s} {puntual[m + '_una']:9.3f} {puntual[m + '_dos']:10.3f}   "
                  f"{v['dos_menos_una']:+.3f} [{v['ic95'][0]:+.3f}, {v['ic95'][1]:+.3f}]{' *' if v['significativa'] else ''}")
        print(f"Acierto de categoria con UNA cara anotada (sin error del modelo): "
              f"{puntual['ocde_acierto_una_cara_real']:.1%}")
    else:
        print(f"\nSolo {len(completos)} pares tienen ambos lados anotados: faltan las etiquetas del lado faltante "
              f"en {args.labels_dos_caras} para medir el acierto por fruto.")

    with open(os.path.join(args.out, "metricas.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(f"\nResultados en {args.out}")


if __name__ == "__main__":
    main()
