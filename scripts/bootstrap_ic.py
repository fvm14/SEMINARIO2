"""
bootstrap_ic.py
Intervalos de confianza al 95% por bootstrap para las metricas por imagen de
una evaluacion (predicciones.csv de eval_yolo_multitarea.py).

El remuestreo se hace POR FRUTO (bootstrap por conglomerados): todas las fotos
de una misma palta entran o salen juntas, porque estan correlacionadas; hacerlo
por imagen daria intervalos artificialmente estrechos.

Con --comparar, calcula ademas el IC de la diferencia (A - B) entre dos modelos
evaluados sobre las mismas imagenes (bootstrap pareado): si el intervalo
incluye 0, la diferencia no es distinguible del ruido.

Nota: mAP de Ultralytics no se incluye porque se calcula a nivel de dataset.

Uso:
    python scripts/bootstrap_ic.py --eval resultados/yolo_multitarea/yolov8s_mt_peso2/eval_test_...
    python scripts/bootstrap_ic.py --eval <dir A> --comparar <dir B>
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CATS = ["Categoria I", "Categoria II", "Rechazado"]


def fruto(archivo):
    p = archivo.split("_")
    return f"{p[0]}_{p[2]}"


def metricas(d):
    mr, mp = d.madurez_real.values, d.madurez_pred.values
    rech = d[d.ocde_real == "Rechazado"]
    return {
        "madurez_accuracy": float((mr == mp).mean()),
        "madurez_f1_macro": float(f1_score(mr, mp, labels=[1, 2, 3, 4, 5], average="macro", zero_division=0)),
        "madurez_accuracy_pm1": float((np.abs(mr - mp) <= 1).mean()),
        "ocde_accuracy": float((d.ocde_real == d.ocde_pred).mean()),
        "ocde_f1_macro": float(f1_score(d.ocde_real, d.ocde_pred, labels=CATS, average="macro", zero_division=0)),
        "recall_rechazado": float((rech.ocde_pred == "Rechazado").mean()) if len(rech) else float("nan"),
        "mae_ratio": float((d.ratio_real - d.ratio_pred).abs().mean()),
    }


def bootstrap(d, n, rng, d2=None):
    grupos = d.groupby("fruto").indices
    claves = list(grupos)
    idx_por_fruto = [grupos[k] for k in claves]
    muestras = []
    for _ in range(n):
        elegidos = rng.integers(0, len(claves), len(claves))
        idx = np.concatenate([idx_por_fruto[i] for i in elegidos])
        m = metricas(d.iloc[idx])
        if d2 is not None:
            m2 = metricas(d2.iloc[idx])
            m = {k: m[k] - m2[k] for k in m}
        muestras.append(m)
    return pd.DataFrame(muestras)


def cargar(ruta):
    d = pd.read_csv(os.path.join(ruta, "predicciones.csv"))
    d["fruto"] = d.archivo.map(fruto)
    return d.sort_values("archivo").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--comparar", default="")
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    d = cargar(args.eval)
    puntual = metricas(d)
    bs = bootstrap(d, args.n, rng)
    res = {"n_imagenes": len(d), "n_frutos": int(d.fruto.nunique()), "n_bootstrap": args.n, "metricas": {}}
    print(f"{args.eval}\n{len(d)} imagenes, {d.fruto.nunique()} frutos, {args.n} remuestreos por fruto\n")
    print(f"{'metrica':24s} {'valor':>8s}   IC 95%")
    for k, v in puntual.items():
        lo, hi = np.nanpercentile(bs[k], [2.5, 97.5])
        res["metricas"][k] = {"valor": round(v, 4), "ic95": [round(lo, 4), round(hi, 4)]}
        print(f"{k:24s} {v:8.4f}   [{lo:.4f}, {hi:.4f}]")

    if args.comparar:
        d2 = cargar(args.comparar)
        assert list(d.archivo) == list(d2.archivo), "las evaluaciones no tienen las mismas imagenes"
        p2 = metricas(d2)
        bsd = bootstrap(d, args.n, np.random.default_rng(args.seed), d2)
        res["comparacion"] = {"contra": args.comparar, "diferencias": {}}
        print(f"\nDiferencia (A - B), B = {args.comparar}")
        print(f"{'metrica':24s} {'A - B':>8s}   IC 95%              distinguible de 0")
        for k in puntual:
            dif = puntual[k] - p2[k]
            lo, hi = np.nanpercentile(bsd[k], [2.5, 97.5])
            sig = not (lo <= 0 <= hi)
            res["comparacion"]["diferencias"][k] = {"diferencia": round(dif, 4), "ic95": [round(lo, 4), round(hi, 4)], "significativa": sig}
            print(f"{k:24s} {dif:+8.4f}   [{lo:+.4f}, {hi:+.4f}]   {'SI' if sig else 'no'}")

    salida = os.path.join(args.eval, "intervalos_confianza.json")
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nGuardado en {salida}")


if __name__ == "__main__":
    main()
