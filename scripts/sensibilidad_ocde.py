"""
Sensibilidad de la categorizacion OCDE al area de referencia del fruto.

Los limites de categoria se obtienen como 4 cm2 / area (Cat. I) y 6 cm2 / area
(Cat. II). Como las imagenes no tienen escala, el area se toma de mediciones
de palta Hass. Este script recalcula la categoria real y la predicha de cada
modelo con varias areas, a partir de los ratios ya guardados en
predicciones.csv (no requiere volver a inferir) y con IC 95% por bootstrap
agrupado por fruto.

Uso:
    python scripts/sensibilidad_ocde.py
Salida: resultados/sensibilidad_ocde/{sensibilidad.csv, sensibilidad.md, sensibilidad.png}
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "resultados" / "sensibilidad_ocde"

MODELOS = {
    "YOLOv8s-seg multitarea (propuesto)": "resultados/yolo_multitarea/yolov8s_mt_ronda1/eval_test_20260924_021049/predicciones.csv",
    "U-Net ResNet34 multitarea": "resultados/comparacion/unet_resnet34/eval_test_20261002_140015/predicciones.csv",
    "YOLOv8s-seg + ResNet-34": "resultados/comparacion/dos_redes/predicciones.csv",
}

# Area proyectada de referencia (cm2): factor de forma 0.975 medido en el dataset
# por largo x diametro reportados para palta Hass.
AREAS = {
    "Jewe, Etiopia (40.0 cm2)": 40.0,
    "Valor actual (42.4 cm2)": 42.4,
    "Upper Gana, Etiopia (46.5 cm2)": 46.5,
    "Mexico, calibre comercial (63.3 cm2)": 63.3,
}
DEF_CAT_I, DEF_CAT_II = 4.0, 6.0
N_BOOT, SEMILLA = 5000, 42


def categoria(ratio, area):
    return np.where(ratio <= DEF_CAT_I / area, 0, np.where(ratio <= DEF_CAT_II / area, 1, 2))


def fruto(archivo):
    # T10_d02_275_a_1_jpg... -> ambiente + id del fruto (T10_275)
    p = archivo.split("_")
    return f"{p[0]}_{p[2]}"


def metricas(real, pred):
    acierto = (real == pred).mean()
    f1 = []
    for c in range(3):
        tp = ((pred == c) & (real == c)).sum()
        fp = ((pred == c) & (real != c)).sum()
        fn = ((pred != c) & (real == c)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan)
    rech = real == 2
    return acierto, np.nanmean(f1), (pred[rech] == 2).mean() if rech.any() else np.nan


def ic_acierto(ok, grupos, rng):
    ids = np.unique(grupos)
    por = {g: ok[grupos == g] for g in ids}
    vals = []
    for _ in range(N_BOOT):
        m = rng.choice(ids, len(ids), replace=True)
        vals.append(np.concatenate([por[g] for g in m]).mean())
    return np.percentile(vals, [2.5, 97.5])


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for modelo, ruta in MODELOS.items():
        df = pd.read_csv(RAIZ / ruta)
        grupos = df.archivo.map(fruto).to_numpy()
        for nombre, area in AREAS.items():
            real = categoria(df.ratio_real.to_numpy(), area)
            pred = categoria(df.ratio_pred.to_numpy(), area)
            acierto, f1, rec = metricas(real, pred)
            lo, hi = ic_acierto((real == pred).astype(float), grupos, rng)
            filas.append({
                "modelo": modelo, "area": nombre, "area_cm2": area,
                "limite_cat_i": DEF_CAT_I / area, "limite_cat_ii": DEF_CAT_II / area,
                "n_cat_i": int((real == 0).sum()), "n_cat_ii": int((real == 1).sum()), "n_rechazado": int((real == 2).sum()),
                "acierto_ocde": acierto, "ic_inf": lo, "ic_sup": hi, "f1_macro_ocde": f1, "recall_rechazado": rec,
            })
    t = pd.DataFrame(filas)
    t.to_csv(SALIDA / "sensibilidad.csv", index=False)

    lineas = ["# Sensibilidad de la categoría OCDE al área de referencia", "",
              "Test: 108 imágenes, 39 frutos. Los límites son 4 cm² / área (Cat. I) y 6 cm² / área (Cat. II); "
              "la categoría real y la predicha se recalculan con cada área. IC 95% por bootstrap agrupado por fruto.", ""]
    ref = t[t.modelo == next(iter(MODELOS))]
    lineas += ["## Límites y distribución real", "", "| Área | Límite Cat. I | Límite Cat. II | Cat. I | Cat. II | Rechazado |", "|---|---|---|---|---|---|"]
    for _, r in ref.iterrows():
        lineas.append(f"| {r.area} | {r.limite_cat_i:.1%} | {r.limite_cat_ii:.1%} | {r.n_cat_i} | {r.n_cat_ii} | {r.n_rechazado} |")
    lineas += ["", "## Acierto OCDE por modelo", "", "| Modelo | " + " | ".join(AREAS) + " |", "|---|" + "---|" * len(AREAS)]
    for m in MODELOS:
        g = t[t.modelo == m]
        lineas.append(f"| {m} | " + " | ".join(f"{r.acierto_ocde:.1%} [{r.ic_inf:.1%}, {r.ic_sup:.1%}]" for _, r in g.iterrows()) + " |")
    for col, tit in (("f1_macro_ocde", "F1 macro OCDE"), ("recall_rechazado", "Recall de Rechazado")):
        lineas += ["", f"## {tit}", "", "| Modelo | " + " | ".join(AREAS) + " |", "|---|" + "---|" * len(AREAS)]
        for m in MODELOS:
            g = t[t.modelo == m]
            lineas.append(f"| {m} | " + " | ".join(f"{v:.3f}" if col == "f1_macro_ocde" else f"{v:.1%}" for v in g[col]) + " |")
    (SALIDA / "sensibilidad.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, mk in zip(MODELOS, "osd"):
        g = t[t.modelo == m]
        ax.errorbar(g.area_cm2, g.acierto_ocde * 100, yerr=[(g.acierto_ocde - g.ic_inf) * 100, (g.ic_sup - g.acierto_ocde) * 100],
                    marker=mk, capsize=3, label=m)
    ax.axvline(42.4, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Área de referencia del fruto (cm²)")
    ax.set_ylabel("Acierto OCDE (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SALIDA / "sensibilidad.png", dpi=200)
    print((SALIDA / "sensibilidad.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
