"""
generar_pseudoetiquetas.py
Etapa 1 de Seminario II: pseudo-etiquetado del dataset completo (14,710 fotos)
con el modelo multitarea final.

1) Manifest completo (data/splits/manifest_completo.csv): cada foto del dataset
   original hereda el split de su fruto. Los 63 frutos que nunca se anotaron se
   reparten 70/20/10 por fruto y balanceados por madurez (StratifiedGroupKFold,
   semilla 42), igual que split_dataset.py. Asi ninguna palta de val/test entra
   a entrenamiento.
2) Pseudo-etiquetas SOLO para fotos de frutos de TRAIN que no tienen anotacion
   humana valida (las 18 de train con anotacion rota tambien se re-etiquetan).
   Formato YOLO-seg igual al de labels_seg (clase 0 palta, clase 1 defecto).
3) Clasificacion de cada foto para la revision manual:
     auto     -> prediccion confiable (una sola palta muy segura, defectos
                 seguros y area plausible)
     revisar  -> algo dudoso (motivo en la columna "motivo")
   Se generan imagenes de revision (anotacion superpuesta) para todas las
   "revisar" y para una muestra al azar de las "auto" (para medir su error).

La madurez NO se pseudo-etiqueta: viene en el nombre del archivo.

Uso (desde la raiz del proyecto):
    python scripts/generar_pseudoetiquetas.py --weights resultados/yolo_multitarea/yolov8s_mt_peso2/weights/best.pt
Salida en data/pseudo/: labels/*.txt, pseudoetiquetas.csv, revision/*.jpg
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_multitarea  # noqa: E402,F401  (necesario para cargar el checkpoint)
from ultralytics import YOLO  # noqa: E402

DIR_CRUDO = os.path.join("data", "raw", "Hass Avocado Ripening Photographic Dataset", "Avocado Ripening Dataset")
SEED = 42


def construir_manifest_completo(ruta_salida):
    fs = sorted(f for f in os.listdir(DIR_CRUDO) if f.lower().endswith(".jpg"))
    d = pd.DataFrame({"archivo": fs})
    p = d.archivo.str[:-4].str.split("_", expand=True)
    d["fruto"] = p[0] + "_" + p[2]
    d["madurez"] = p[4].astype(int)

    m = pd.read_csv(os.path.join("data", "splits", "split_manifest.csv"))
    split_de = dict(zip(m.fruto, m.split))
    d["split"] = d.fruto.map(split_de)

    nuevos = d[d.split.isna()]
    if len(nuevos):
        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
        asignacion = {}
        for k, (_, idx) in enumerate(sgkf.split(nuevos, nuevos.madurez, groups=nuevos.fruto)):
            sp = "train" if k < 7 else ("val" if k < 9 else "test")
            for f in nuevos.iloc[idx].fruto.unique():
                asignacion[f] = sp
        d.loc[d.split.isna(), "split"] = d.loc[d.split.isna(), "fruto"].map(asignacion)
        d["fruto_nuevo"] = d.fruto.isin(asignacion)
    else:
        d["fruto_nuevo"] = False

    # anotacion humana valida (las 1,243 menos las 24 rotas)
    m["orig"] = m.archivo.str.split("_jpg").str[0] + ".jpg"
    excl = set(pd.read_csv(os.path.join("data", "splits", "anotaciones_excluidas.csv")).archivo)
    validas = set(m[~m.archivo.isin(excl)].orig)
    d["anotacion_humana"] = d.archivo.isin(validas)
    d.to_csv(ruta_salida, index=False)
    return d


def poligono_mayor(xy):
    """Ultralytics puede devolver varios poligonos por mascara; se usa el de mayor area."""
    return xy if len(xy) >= 3 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default=os.path.join("data", "pseudo"))
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf-palta", type=float, default=0.35)
    ap.add_argument("--conf-defecto", type=float, default=0.15)
    ap.add_argument("--auto-palta", type=float, default=0.85, help="confianza minima de la palta para 'auto'")
    ap.add_argument("--auto-defecto", type=float, default=0.50, help="un defecto con confianza menor a esto es 'dudoso'")
    ap.add_argument("--max-dudosa", type=float, default=0.50,
                    help="se acepta la foto si el area de defectos dudosos es <= esta fraccion del area de defecto "
                         "(regla elegida en validacion con calibrar_regla_pseudo.py)")
    ap.add_argument("--muestra-auto", type=float, default=0.04, help="fraccion de 'auto' que se revisa como control")
    ap.add_argument("--limite", type=int, default=0, help="solo para pruebas")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "revision"), exist_ok=True)

    d = construir_manifest_completo(os.path.join("data", "splits", "manifest_completo.csv"))
    print("Manifest completo:", d.split.value_counts().to_dict(),
          "| frutos por split:", d.groupby("split").fruto.nunique().to_dict())
    pendientes = d[(d.split == "train") & (~d.anotacion_humana)].reset_index(drop=True)
    if args.limite:
        pendientes = pendientes.sample(args.limite, random_state=SEED).reset_index(drop=True)
    print(f"Fotos de train a pseudo-etiquetar: {len(pendientes)}")

    modelo = YOLO(args.weights)
    rng = np.random.default_rng(SEED)
    filas = []
    for n, fila in pendientes.iterrows():
        ruta = os.path.join(DIR_CRUDO, fila.archivo)
        r = modelo.predict(ruta, imgsz=args.imgsz, conf=min(args.conf_palta, args.conf_defecto),
                           retina_masks=True, verbose=False, **({"device": args.device} if args.device else {}))[0]
        alto, ancho = r.orig_shape
        motivos = []
        lineas = []
        conf_palta, confs_def, area_palta, area_def = 0.0, [], 0, 0
        m_palta = np.zeros((alto, ancho), bool)
        m_def = np.zeros((alto, ancho), bool)
        m_dud = np.zeros((alto, ancho), bool)

        if r.masks is not None and len(r.boxes):
            clases = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            mascaras = r.masks.data.cpu().numpy() > 0.5
            poligonos = r.masks.xyn
            idx_palta = [i for i in range(len(clases)) if clases[i] == 0 and confs[i] >= args.conf_palta]
            if len(idx_palta) > 1:
                motivos.append(f"{len(idx_palta)} paltas detectadas")
            if idx_palta:
                i = max(idx_palta, key=lambda j: confs[j])  # una sola palta por foto
                conf_palta = float(confs[i])
                m_palta = mascaras[i]
                if len(poligonos[i]) >= 3:
                    lineas.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poligonos[i]))
            for i in range(len(clases)):
                if clases[i] == 1 and confs[i] >= args.conf_defecto and len(poligonos[i]) >= 3:
                    # ROI: el defecto tiene que caer mayormente sobre la palta
                    if m_palta.any() and (mascaras[i] & m_palta).sum() < 0.5 * mascaras[i].sum():
                        continue
                    cnts, _ = cv2.findContours(mascaras[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(c)
                        if w * h > 400 and cv2.contourArea(c) / (w * h) > 0.9:
                            motivos.append("defecto rectangular")
                    confs_def.append(float(confs[i]))
                    m_def |= mascaras[i]
                    if confs[i] < args.auto_defecto:
                        m_dud |= mascaras[i]
                    lineas.append("1 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poligonos[i]))

        area_palta = m_palta.sum() / (alto * ancho)
        ratio = m_def.sum() / max(1, m_palta.sum())
        if conf_palta == 0:
            motivos.append("sin palta")
        elif conf_palta < args.auto_palta:
            motivos.append(f"palta poco segura ({conf_palta:.2f})")
        if 0 < area_palta < 0.025:
            motivos.append("palta diminuta")
        frac_dudosa = (m_dud & m_def).sum() / max(1, m_def.sum())
        if frac_dudosa > args.max_dudosa:
            motivos.append(f"area de defecto dudosa ({frac_dudosa*100:.0f}%)")
        if ratio > 1:
            motivos.append("defecto mayor que la palta")
        estado = "revisar" if motivos else "auto"

        stem = fila.archivo[:-4]
        with open(os.path.join(args.out, "labels", stem + ".txt"), "w") as f:
            f.write("\n".join(lineas) + ("\n" if lineas else ""))

        revisar_img = estado == "revisar" or rng.random() < args.muestra_auto
        if revisar_img:
            o = r.orig_img.astype(np.float32).copy()
            o[m_palta] = o[m_palta] * 0.8 + np.array([0, 200, 0]) * 0.2
            o[m_def] = o[m_def] * 0.55 + np.array([0, 0, 255]) * 0.45
            o = o.astype(np.uint8)
            for c, color in ((m_palta, (0, 170, 0)), (m_def, (0, 0, 255))):
                cnts, _ = cv2.findContours(c.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(o, cnts, -1, color, 2)
            sep = np.full((alto, 8, 3), 255, np.uint8)
            lienzo = np.concatenate([r.orig_img, sep, o], axis=1)  # original | pseudo-etiqueta
            lienzo = cv2.resize(lienzo, (lienzo.shape[1] * 3 // 4, lienzo.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(args.out, "revision", stem + ".jpg"), lienzo, [cv2.IMWRITE_JPEG_QUALITY, 85])

        filas.append({"archivo": fila.archivo, "fruto": fila.fruto, "madurez": fila.madurez,
                      "estado": estado, "motivo": "; ".join(motivos), "en_revision": revisar_img,
                      "conf_palta": round(conf_palta, 3), "n_defectos": len(confs_def),
                      "conf_defecto_min": round(min(confs_def), 3) if confs_def else "",
                      "ratio_defecto": round(float(ratio), 4), "frac_dudosa": round(float(frac_dudosa), 3)})
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(pendientes)}")

    res = pd.DataFrame(filas)
    res.to_csv(os.path.join(args.out, "pseudoetiquetas.csv"), index=False)
    print("\n==== RESUMEN ====")
    print(res.estado.value_counts().to_string())
    print(f"Imagenes para revisar: {int(res.en_revision.sum())} "
          f"({int((res.estado == 'revisar').sum())} dudosas + {int(((res.estado == 'auto') & res.en_revision).sum())} muestra de auto)")
    motivos = res[res.estado == "revisar"].motivo.str.split("; ").explode().str.replace(r"\s*\(.*\)", "", regex=True)
    print("Motivos:\n" + motivos.str.replace(r"^\d+ ", "", regex=True).value_counts().to_string())


if __name__ == "__main__":
    main()
