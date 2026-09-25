"""
inferencia_movil.py
Inferencia y post-proceso de referencia para los modelos exportados (ONNX o
TFLite), escrito solo con NumPy/OpenCV: es la especificacion de lo que la app
movil debe implementar (Etapa 4), paso a paso.

Pasos:
  1. letterbox: redimensionar manteniendo proporcion y rellenar hasta HxW (gris 114)
  2. inferencia -> detecciones [38, N], prototipos [32, H/4, W/4], madurez [5]
  3. filtrar por confianza (palta >= 0.35, defecto >= umbral calibrado)
  4. NMS por clase (IoU 0.7)
  5. mascara por instancia = sigmoide(coeficientes @ prototipos), recortada a su caja
  6. volver a coordenadas de la foto original
  7. fruto completo, filtro ROI, ratio defecto/palta, categoria OCDE
  8. madurez = argmax de las 5 probabilidades

Tambien evalua un modelo exportado sobre val/test (mismas metricas que
eval_yolo_multitarea.py) y mide la latencia de la inferencia.

Uso:
    python scripts/inferencia_movil.py --modelo modelos_movil/palta_multitarea_800.onnx --conf-defecto 0.05
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocde import (calcular_ratio, clasificar_ocde, filtrar_defecto_por_roi,  # noqa: E402
                  fruto_completo, CATEGORIAS_OCDE)
from rasterize_utils import poligonos_a_mascaras  # noqa: E402


# ---------------------------------------------------------------- backends
class BackendONNX:
    def __init__(self, ruta, hilos=0):
        import onnxruntime as ort
        op = ort.SessionOptions()
        if hilos:
            op.intra_op_num_threads = hilos
        self.s = ort.InferenceSession(ruta, op, providers=["CPUExecutionProvider"])
        self.entrada = self.s.get_inputs()[0].name
        self.imgsz = self.s.get_inputs()[0].shape[2]

    def __call__(self, x_nchw):
        o = self.s.run(None, {self.entrada: x_nchw})
        if len(o) == 3:  # formato antiguo: detecciones juntas
            return o[0][0], o[1][0], o[2][0]
        cajas, punt, coef, proto, mad = (a[0] for a in o)
        return np.concatenate([cajas * self.imgsz, punt, coef], 0), proto, mad


class BackendTFLite:
    def __init__(self, ruta, hilos=0):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        self.it = Interpreter(model_path=ruta, num_threads=hilos or None)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]
        self.outs = self.it.get_output_details()
        self.imgsz = self.inp["shape"][1]  # NHWC

    def _leer(self, det):
        a = self.it.get_tensor(det["index"])
        if det["dtype"] != np.float32:  # salida cuantizada
            esc, cero = det["quantization"]
            a = (a.astype(np.float32) - cero) * esc
        return a

    def __call__(self, x_nchw):
        x = x_nchw.transpose(0, 2, 3, 1)
        if self.inp["dtype"] != np.float32:
            esc, cero = self.inp["quantization"]
            x = np.clip(np.round(x / esc + cero), -128, 127).astype(self.inp["dtype"])
        self.it.set_tensor(self.inp["index"], x)
        self.it.invoke()
        det = proto = mad = None
        partes = {}
        for o in self.outs:
            a = self._leer(o)[0]
            if a.size == 5:
                mad = a.reshape(5)
            elif a.ndim == 3:  # prototipos
                proto = a.transpose(2, 0, 1) if a.shape[-1] == 32 else a
            else:  # 2D: [C, N] o [N, C]
                a = a if a.shape[0] < a.shape[1] else a.T
                partes[a.shape[0]] = a
        if 38 in partes:  # formato antiguo
            det = partes[38]
        else:
            det = np.concatenate([partes[4] * self.imgsz, partes[2], partes[32]], 0)
        return det, proto, mad


def cargar_backend(ruta, hilos=0):
    return BackendTFLite(ruta, hilos) if ruta.endswith(".tflite") else BackendONNX(ruta, hilos)


# ---------------------------------------------------------------- pre/post-proceso
def letterbox(img_bgr, imgsz):
    h, w = img_bgr.shape[:2]
    r = min(imgsz / h, imgsz / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    lienzo = np.full((imgsz, imgsz, 3), 114, np.uint8)
    lienzo[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x = cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return x, r, (top, left, nh, nw)


def nms(cajas, puntajes, iou=0.7):
    x1, y1, x2, y2 = cajas.T
    areas = (x2 - x1) * (y2 - y1)
    orden = puntajes.argsort()[::-1]
    keep = []
    while orden.size:
        i = orden[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[orden[1:]]); yy1 = np.maximum(y1[i], y1[orden[1:]])
        xx2 = np.minimum(x2[i], x2[orden[1:]]); yy2 = np.minimum(y2[i], y2[orden[1:]])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        orden = orden[1:][inter / (areas[i] + areas[orden[1:]] - inter + 1e-9) <= iou]
    return np.array(keep, dtype=int)


def postprocesar(det, proto, forma_orig, r, pad, conf_palta=0.35, conf_defecto=0.15, iou=0.7):
    """Devuelve (mascara_palta, mascara_defecto) booleanas en la resolucion de la foto original."""
    H0, W0 = forma_orig
    top, left, nh, nw = pad
    cxcywh, puntajes, coefs = det[:4].T, det[4:6].T, det[6:].T
    clase = puntajes.argmax(1)
    conf = puntajes.max(1)
    umbral = np.where(clase == 0, conf_palta, conf_defecto)
    sel = conf >= umbral
    cxcywh, clase, conf, coefs = cxcywh[sel], clase[sel], conf[sel], coefs[sel]
    palta = np.zeros((H0, W0), bool)
    defecto = np.zeros((H0, W0), bool)
    if not len(conf):
        return palta, defecto
    cajas = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                      cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], 1)
    keep = nms(cajas + clase[:, None] * 4096.0, conf, iou)[:300]  # NMS por clase
    c, h, w = proto.shape
    S = proto.shape[1] * 4  # tamano de entrada
    for k in keep:
        m = 1 / (1 + np.exp(-(coefs[k] @ proto.reshape(c, -1)))).reshape(h, w)
        m = cv2.resize(m, (S, S), interpolation=cv2.INTER_LINEAR)
        x1, y1, x2, y2 = np.clip(cajas[k], 0, S).astype(int)
        recorte = np.zeros_like(m, dtype=bool)
        recorte[y1:y2, x1:x2] = m[y1:y2, x1:x2] > 0.5
        recorte = recorte[top:top + nh, left:left + nw]
        recorte = cv2.resize(recorte.astype(np.uint8), (W0, H0), interpolation=cv2.INTER_NEAREST).astype(bool)
        if clase[k] == 0:
            palta |= recorte
        else:
            defecto |= recorte
    return palta, defecto


def analizar(backend, img_bgr, conf_palta=0.35, conf_defecto=0.15):
    """Pipeline completo de la app: foto -> (categoria OCDE, ratio, madurez 1-5, prob, mascaras, ms)."""
    x, r, pad = letterbox(img_bgr, backend.imgsz)
    t0 = time.perf_counter()
    det, proto, mad = backend(x)
    ms = (time.perf_counter() - t0) * 1000
    palta, defecto = postprocesar(det, proto, img_bgr.shape[:2], r, pad, conf_palta, conf_defecto)
    fruto = fruto_completo(palta, defecto)
    defecto = filtrar_defecto_por_roi(fruto, defecto)
    ratio = calcular_ratio(fruto, defecto)
    return {"ocde": CATEGORIAS_OCDE[clasificar_ocde(ratio)], "ratio": ratio, "madurez": int(mad.argmax()) + 1,
            "prob_madurez": float(mad.max()), "fruto": fruto, "defecto": defecto, "ms": ms}


# ---------------------------------------------------------------- evaluacion
def madurez_de(nombre):
    base = os.path.splitext(os.path.basename(nombre))[0]
    return int(base.split("_")[4].split(".")[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--data", default=os.path.join("data", "yolo", "data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--labels-seg", default=os.path.join("data", "raw", "labels_seg"))
    ap.add_argument("--conf-defecto", type=float, default=0.15)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--max-imagenes", type=int, default=0)
    ap.add_argument("--salida", default="")
    args = ap.parse_args()

    raiz = os.path.dirname(os.path.abspath(args.data))
    with open(os.path.join(raiz, f"{args.split}.txt"), encoding="utf-8") as f:
        imgs = [l.strip() for l in f if l.strip()]
    if args.max_imagenes:
        imgs = imgs[:args.max_imagenes]
    backend = cargar_backend(args.modelo, args.hilos)

    tp = np.zeros(2); fp = np.zeros(2); fn = np.zeros(2)
    mr, mp, og, op, mae, lat = [], [], [], [], [], []
    for ruta in imgs:
        img = cv2.imread(ruta)
        res = analizar(backend, img, 0.35, args.conf_defecto)
        stem = os.path.splitext(os.path.basename(ruta))[0]
        gp, gd = poligonos_a_mascaras(os.path.join(args.labels_seg, stem + ".txt"), *img.shape[:2])
        gp, gd = gp.astype(bool), gd.astype(bool)
        for i, (pr, gt) in enumerate(((res["fruto"], gp), (res["defecto"], gd))):
            tp[i] += (pr & gt).sum(); fp[i] += (pr & ~gt).sum(); fn[i] += (~pr & gt).sum()
        r_gt = calcular_ratio(gp, gd)
        og.append(CATEGORIAS_OCDE[clasificar_ocde(r_gt)]); op.append(res["ocde"])
        mae.append(abs(r_gt - res["ratio"]))
        mr.append(madurez_de(ruta)); mp.append(res["madurez"]); lat.append(res["ms"])

    iou = tp / (tp + fp + fn + 1e-9)
    mr, mp = np.array(mr), np.array(mp)
    lat = np.array(lat[3:] if len(lat) > 5 else lat)
    og, op = np.array(og), np.array(op)
    rech = og == "Rechazado"
    out = {
        "modelo": args.modelo, "tamano_mb": round(os.path.getsize(args.modelo) / 1e6, 2), "imgsz": int(backend.imgsz),
        "n": len(imgs), "conf_defecto": args.conf_defecto,
        "iou_palta": round(float(iou[0]), 4), "iou_defecto": round(float(iou[1]), 4),
        "precision_defecto": round(float(tp[1] / (tp[1] + fp[1] + 1e-9)), 4),
        "recall_defecto": round(float(tp[1] / (tp[1] + fn[1] + 1e-9)), 4),
        "madurez_acc": round(float((mr == mp).mean()), 4),
        "madurez_f1": round(float(f1_score(mr, mp, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0)), 4),
        "ocde_acc": round(float((og == op).mean()), 4),
        "recall_rechazado": round(float((op[rech] == "Rechazado").mean()), 4) if rech.any() else None,
        "mae_ratio": round(float(np.mean(mae)), 4),
        "latencia_ms_media": round(float(lat.mean()), 1), "latencia_ms_p95": round(float(np.percentile(lat, 95)), 1),
        "hilos_cpu": args.hilos,
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    if args.salida:
        with open(args.salida, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
