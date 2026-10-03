"""
map_movil.py
mAP50 y mAP50-95 (cajas y mascaras, clases palta y defecto) de las variantes
exportadas del modelo multitarea (ONNX FP32, TFLite FP16, rango dinamico e
INT8), para medir cuanto cambia la deteccion al cuantizar.

Mismo criterio que la validacion de Ultralytics: confianza minima 0.001, NMS
por clase con IoU 0.7, hasta 300 detecciones, emparejamiento por clase en IoU
0.50:0.95 y AP por clase con ap_per_class. Las mascaras se comparan a la
resolucion de los prototipos (1/4 de la entrada), como Ultralytics. Todas las
variantes pasan por el mismo letterbox (scripts/inferencia_movil.py), asi que
la diferencia entre ellas se debe solo a la cuantizacion.

Uso:
    python scripts/map_movil.py modelos_movil/palta_multitarea_800.onnx \
        modelos_movil/palta_multitarea_800_float16.tflite --salida resultados/etapa2/map_variantes.json
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia_movil import cargar_backend, letterbox, nms  # noqa: E402
from ultralytics.utils.metrics import ap_per_class  # noqa: E402

IOUV = np.linspace(0.5, 0.95, 10)


def etiquetas(ruta_img, forma, r, pad, lado_proto):
    """Poligonos de la etiqueta YOLO -> clases, cajas (px del lienzo) y mascaras (resolucion de prototipos)."""
    h0, w0 = forma
    top, left = pad[0], pad[1]
    stem = os.path.splitext(os.path.basename(ruta_img))[0]
    ruta = os.path.join(os.path.dirname(os.path.dirname(ruta_img)), "labels", stem + ".txt")
    clases, cajas, mascaras = [], [], []
    escala = lado_proto / (lado_proto * 4)
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            v = linea.split()
            if len(v) < 7:
                continue
            p = np.array(v[1:], np.float32).reshape(-1, 2)
            p[:, 0] = p[:, 0] * w0 * r + left
            p[:, 1] = p[:, 1] * h0 * r + top
            m = np.zeros((lado_proto, lado_proto), np.uint8)
            cv2.fillPoly(m, [np.round(p * escala).astype(np.int32)], 1)
            clases.append(int(v[0]))
            cajas.append([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()])
            mascaras.append(m.astype(bool).ravel())
    return np.array(clases), np.array(cajas, np.float32).reshape(-1, 4), np.array(mascaras).reshape(len(clases), -1)


def predicciones(det, proto, conf_min=0.001, iou=0.7, max_det=300):
    cxcywh, punt, coefs = det[:4].T, det[4:6].T, det[6:].T
    clase, conf = punt.argmax(1), punt.max(1)
    sel = conf >= conf_min
    cxcywh, clase, conf, coefs = cxcywh[sel], clase[sel], conf[sel], coefs[sel]
    cajas = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                      cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], 1)
    keep = nms(cajas + clase[:, None] * 4096.0, conf, iou)[:max_det]
    cajas, clase, conf, coefs = cajas[keep], clase[keep], conf[keep], coefs[keep]
    c, h, w = proto.shape
    m = 1 / (1 + np.exp(-(coefs @ proto.reshape(c, -1)))) > 0.5
    m = m.reshape(-1, h, w)
    # recorte a la caja, en la escala de los prototipos
    ys, xs = np.arange(h)[None, :, None], np.arange(w)[None, None, :]
    b = cajas / 4
    dentro = (xs >= b[:, 0, None, None]) & (xs < b[:, 2, None, None]) & (ys >= b[:, 1, None, None]) & (ys < b[:, 3, None, None])
    return clase, conf, cajas, (m & dentro).reshape(len(clase), -1)


def iou_cajas(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def iou_mascaras(a, b):
    a, b = a.astype(np.float32), b.astype(np.float32)
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None, :] - inter + 1e-9)


def emparejar(clase_pred, clase_gt, iou):
    """Igual que BaseValidator.match_predictions de Ultralytics (iou: gt x pred)."""
    correcto = np.zeros((len(clase_pred), len(IOUV)), bool)
    iou = iou * (clase_gt[:, None] == clase_pred[None, :])
    for i, t in enumerate(IOUV):
        pares = np.array(np.nonzero(iou >= t)).T
        if len(pares):
            if len(pares) > 1:
                pares = pares[iou[pares[:, 0], pares[:, 1]].argsort()[::-1]]
                pares = pares[np.unique(pares[:, 1], return_index=True)[1]]
                pares = pares[np.unique(pares[:, 0], return_index=True)[1]]
            correcto[pares[:, 1], i] = True
    return correcto


def evaluar(ruta_modelo, imgs):
    backend = cargar_backend(ruta_modelo, 4)
    tp_caja, tp_masc, confs, clases_pred, clases_gt = [], [], [], [], []
    for ruta in imgs:
        img = cv2.imread(ruta)
        x, r, pad = letterbox(img, backend.imgsz)
        det, proto, _ = backend(x)
        cp, conf, cajas, masc = predicciones(det, proto)
        cg, cajas_gt, masc_gt = etiquetas(ruta, img.shape[:2], r, pad, proto.shape[1])
        clases_gt.append(cg)
        if len(cp) == 0:
            continue
        tp_caja.append(emparejar(cp, cg, iou_cajas(cajas_gt, cajas)) if len(cg) else np.zeros((len(cp), 10), bool))
        tp_masc.append(emparejar(cp, cg, iou_mascaras(masc_gt, masc)) if len(cg) else np.zeros((len(cp), 10), bool))
        confs.append(conf)
        clases_pred.append(cp)
    conf, cp, cg = np.concatenate(confs), np.concatenate(clases_pred), np.concatenate(clases_gt)
    out = {"modelo": ruta_modelo, "n": len(imgs)}
    for nombre, tp in (("caja", np.concatenate(tp_caja)), ("mascara", np.concatenate(tp_masc))):
        ap = ap_per_class(tp, conf, cp, cg)[5]  # (clases, 10)
        out[f"{nombre}_map50"] = round(float(ap[:, 0].mean()), 4)
        out[f"{nombre}_map50_95"] = round(float(ap.mean()), 4)
        out[f"{nombre}_ap50_palta"] = round(float(ap[0, 0]), 4)
        out[f"{nombre}_ap50_defecto"] = round(float(ap[1, 0]), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("modelos", nargs="+")
    ap.add_argument("--data", default=os.path.join("data", "yolo"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--salida", default="")
    args = ap.parse_args()
    with open(os.path.join(args.data, f"{args.split}.txt"), encoding="utf-8") as f:
        imgs = [l.strip() for l in f if l.strip()]
    resultados = []
    for m in args.modelos:
        r = evaluar(m, imgs)
        print(json.dumps(r, ensure_ascii=False))
        resultados.append(r)
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            json.dump(resultados, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
