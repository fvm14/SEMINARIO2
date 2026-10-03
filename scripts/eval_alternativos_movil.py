"""
eval_alternativos_movil.py
Evalua en TFLite los modelos de la comparacion (Etapa 3), con el mismo
protocolo que sus versiones en PyTorch, para medir lo que se pierde al
cuantizarlos y dejar la referencia con la que se compara la app:

  unet       U-Net ResNet34 multitarea: mascaras de palta (> 0.5) y defecto (> umbral)
  dos_redes  YOLOv8s-seg sin madurez (post-proceso de inferencia_movil.py) + ResNet-34 de madurez

Despues de las mascaras, todo es igual que en el multitarea: fruto completo,
filtro ROI, ratio defecto/fruto y categoria OCDE.

Salida en --salida: metricas.json (mismas claves que inferencia_movil.py),
predicciones.csv (mismo formato que los demas modelos, sirve para bootstrap_ic.py)
y referencia_app.json (por foto, para comparar con la app).

Uso:
    python scripts/eval_alternativos_movil.py --tipo unet --modelo modelos_movil/unet_resnet34_800_dynamic_range_quant.tflite --conf-defecto 0.10 --salida resultados/etapa3_movil/unet_dynamic_range
    python scripts/eval_alternativos_movil.py --tipo dos_redes --modelo modelos_movil/yolov8s_seg_800_dynamic_range_quant.tflite ^
        --clasificador modelos_movil/resnet34_madurez_448_dynamic_range_quant.tflite --conf-defecto 0.10 --salida resultados/etapa3_movil/dos_redes_dynamic_range
    # comparar con lo que obtuvo la app en el celular:
    python scripts/eval_alternativos_movil.py --comparar resultados/etapa3_movil/unet_dynamic_range/referencia_app.json resultados/app/resultados_app_unet.json
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia_movil import BackendTFLite, letterbox, madurez_de, postprocesar  # noqa: E402
from ocde import CATEGORIAS_OCDE, calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi, fruto_completo  # noqa: E402
from rasterize_utils import poligonos_a_mascaras  # noqa: E402


class TFLiteSimple:
    """Interprete con entrada NHWC float 0-1; devuelve las salidas por su tamano."""

    def __init__(self, ruta, hilos=4):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        self.it = Interpreter(model_path=ruta, num_threads=hilos)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()
        self.imgsz = int(self.inp["shape"][1])

    def __call__(self, x_nchw):
        self.it.set_tensor(self.inp["index"], np.ascontiguousarray(x_nchw.transpose(0, 2, 3, 1)))
        self.it.invoke()
        mad, masc = None, None
        for o in self.outs:
            a = self.it.get_tensor(o["index"])[0]
            if a.size == 5:
                mad = a.reshape(5)
            else:  # mascaras: [H, W, 2] (NHWC) o [2, H, W]
                masc = a.transpose(2, 0, 1) if a.shape[-1] == 2 else a
        return masc, mad


def analizar_unet(modelo, img, conf_defecto):
    x, _, (top, left, nh, nw) = letterbox(img, modelo.imgsz)
    t0 = time.perf_counter()
    prob, mad = modelo(x)
    ms = (time.perf_counter() - t0) * 1000
    alto, ancho = img.shape[:2]
    prob = prob[:, top:top + nh, left:left + nw]
    if (nh, nw) != (alto, ancho):
        prob = np.stack([cv2.resize(c, (ancho, alto), interpolation=cv2.INTER_LINEAR) for c in prob])
    return prob[0] > 0.5, prob[1] > conf_defecto, mad, ms


def analizar_dos_redes(seg, clas, img, conf_defecto):
    x, r, pad = letterbox(img, seg.imgsz)
    t0 = time.perf_counter()
    det, proto, _ = seg(x)
    ms_seg = (time.perf_counter() - t0) * 1000
    palta, defecto = postprocesar(det, proto, img.shape[:2], r, pad, 0.35, conf_defecto)
    xc, _, _ = letterbox(img, clas.imgsz)
    t1 = time.perf_counter()
    _, mad = clas(xc)
    ms_clas = (time.perf_counter() - t1) * 1000
    return palta, defecto, mad, ms_seg + ms_clas


def comparar(ref_json, app_json):
    ref = {r["foto"]: r for r in json.load(open(ref_json, encoding="utf-8"))}
    app = {r["foto"]: r for r in json.load(open(app_json, encoding="utf-8"))}
    comunes = sorted(set(ref) & set(app))
    cat = sum(ref[f]["categoria"] == app[f]["categoria"] for f in comunes)
    mad = sum(ref[f]["madurez"] == app[f]["madurez"] for f in comunes)
    dr = np.array([abs(ref[f]["ratio"] - app[f]["ratio"]) for f in comunes])
    dp = np.array([abs(ref[f]["prob_madurez"] - app[f]["prob_madurez"]) for f in comunes])
    lat = {k: np.mean([app[f][k] for f in comunes]) for k in ("ms_pre", "ms_inf", "ms_post")}
    print(f"Fotos comparadas: {len(comunes)}")
    print(f"Categoria OCDE igual: {cat}/{len(comunes)} | madurez igual: {mad}/{len(comunes)}")
    print(f"Ratio: diferencia mediana {np.median(dr):.4f}, maxima {dr.max():.4f}, iguales {(dr < 1e-3).mean():.0%}")
    print(f"Probabilidad de madurez: diferencia maxima {dp.max():.4f}")
    print(f"Latencia en la app: pre {lat['ms_pre']:.0f} ms, inferencia {lat['ms_inf']:.0f} ms, post {lat['ms_post']:.0f} ms, "
          f"total {sum(lat.values()):.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tipo", choices=["unet", "dos_redes"])
    ap.add_argument("--modelo")
    ap.add_argument("--clasificador", default="")
    ap.add_argument("--conf-defecto", type=float, default=0.10)
    ap.add_argument("--data", default=os.path.join("data", "yolo"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--max-imagenes", type=int, default=0)
    ap.add_argument("--salida", default="")
    ap.add_argument("--comparar", nargs=2, metavar=("REFERENCIA", "APP"))
    args = ap.parse_args()
    if args.comparar:
        return comparar(*args.comparar)

    with open(os.path.join(args.data, f"{args.split}.txt"), encoding="utf-8") as f:
        nombres = [l.strip().replace("\\", "/").split("/")[-1] for l in f if l.strip()]
    rutas = [os.path.join(args.data, "images", n) for n in nombres][: args.max_imagenes or None]
    if args.tipo == "unet":
        modelo = TFLiteSimple(args.modelo, args.hilos)
        tam = os.path.getsize(args.modelo)
        analizar = lambda img: analizar_unet(modelo, img, args.conf_defecto)  # noqa: E731
    else:
        seg, clas = BackendTFLite(args.modelo, args.hilos), TFLiteSimple(args.clasificador, args.hilos)
        tam = os.path.getsize(args.modelo) + os.path.getsize(args.clasificador)
        analizar = lambda img: analizar_dos_redes(seg, clas, img, args.conf_defecto)  # noqa: E731

    tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2)
    filas, ref_app, mr, mp, og, op, mae, lat = [], [], [], [], [], [], [], []
    for ruta in rutas:
        img = cv2.imread(ruta)
        palta, defecto, mad, ms = analizar(img)
        fruto = fruto_completo(palta, defecto)
        defecto = filtrar_defecto_por_roi(fruto, defecto)
        ratio = calcular_ratio(fruto, defecto)
        stem = os.path.splitext(os.path.basename(ruta))[0]
        gp, gd = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), *img.shape[:2])
        gp, gd = gp.astype(bool), gd.astype(bool)
        for i, (pr, gt) in enumerate(((fruto, gp), (defecto, gd))):
            tp[i] += (pr & gt).sum(); fp[i] += (pr & ~gt).sum(); fn[i] += (~pr & gt).sum()
        r_gt = calcular_ratio(gp, gd)
        c_gt, c_pr = clasificar_ocde(r_gt), clasificar_ocde(ratio)
        m_gt, m_pr = madurez_de(ruta), int(mad.argmax()) + 1
        og.append(c_gt); op.append(c_pr); mae.append(abs(r_gt - ratio)); mr.append(m_gt); mp.append(m_pr); lat.append(ms)
        filas.append({"archivo": os.path.basename(ruta), "madurez_real": m_gt, "madurez_pred": m_pr,
                      "confianza_madurez": round(float(mad.max()), 4), "ratio_real": round(r_gt, 5),
                      "ratio_pred": round(ratio, 5), "ocde_real": CATEGORIAS_OCDE[c_gt],
                      "ocde_pred": CATEGORIAS_OCDE[c_pr], "latencia_ms": round(ms, 2)})
        ref_app.append({"foto": os.path.basename(ruta), "categoria": c_pr, "ratio": round(ratio, 5), "madurez": m_pr,
                        "prob_madurez": float(mad.max()), "px_fruto": int(fruto.sum()), "px_defecto": int(defecto.sum())})

    iou = tp / (tp + fp + fn + 1e-9)
    mr, mp, og, op = map(np.array, (mr, mp, og, op))
    lat = np.array(lat[3:] if len(lat) > 5 else lat)
    rech = og == 2
    out = {
        "tipo": args.tipo, "modelo": args.modelo, "clasificador": args.clasificador or None,
        "tamano_mb": round(tam / 1e6, 2), "n": len(rutas), "conf_defecto": args.conf_defecto,
        "iou_palta": round(float(iou[0]), 4), "iou_defecto": round(float(iou[1]), 4),
        "precision_defecto": round(float(tp[1] / (tp[1] + fp[1] + 1e-9)), 4),
        "recall_defecto": round(float(tp[1] / (tp[1] + fn[1] + 1e-9)), 4),
        "madurez_acc": round(float((mr == mp).mean()), 4),
        "madurez_f1": round(float(f1_score(mr, mp, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0)), 4),
        "ocde_acc": round(float((og == op).mean()), 4),
        "ocde_f1": round(float(f1_score(og, op, average="macro", labels=[0, 1, 2], zero_division=0)), 4),
        "recall_rechazado": round(float((op[rech] == 2).mean()), 4) if rech.any() else None,
        "mae_ratio": round(float(np.mean(mae)), 4),
        "latencia_ms_media": round(float(lat.mean()), 1), "latencia_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "hilos_cpu": args.hilos,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.salida:
        os.makedirs(args.salida, exist_ok=True)
        with open(os.path.join(args.salida, "metricas.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        with open(os.path.join(args.salida, "predicciones.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        with open(os.path.join(args.salida, "referencia_app.json"), "w", encoding="utf-8") as f:
            json.dump(ref_app, f, indent=1)


if __name__ == "__main__":
    main()
