"""
exportar_movil.py
Etapa 2: exporta el modelo multitarea a ONNX con CINCO salidas, para llevarlo
a movil (ONNX -> TFLite/LiteRT) sin perder la madurez:

  entrada  "imagen"        float32 [1, 3, H, W]  RGB, valores 0-1, letterbox a HxW
  salida   "cajas"         float32 [1, 4, N]   cx, cy, w, h NORMALIZADOS 0-1 (multiplicar por H)
  salida   "puntajes"      float32 [1, 2, N]   confianza palta y defecto por ancla
  salida   "coeficientes"  float32 [1, 32, N]  coeficientes de mascara por ancla
  salida   "prototipos"    float32 [1, 32, H/4, W/4]  prototipos de mascara
  salida   "madurez"       float32 [1, 5]      probabilidades de madurez 1..5 (softmax)

Las salidas de deteccion van SEPARADAS y las cajas normalizadas: si van juntas
en un solo tensor (como exporta Ultralytics), la cuantizacion INT8 usa una sola
escala para valores 0-800 (cajas) y 0-1 (confianzas), y las confianzas quedan en
cero. El exportador estandar de Ultralytics tampoco exporta la madurez.

Uso:
    python scripts/exportar_movil.py --weights resultados/yolo_multitarea/yolov8s_mt_ronda1/weights/best.pt --imgsz 800
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yolo_multitarea  # noqa: E402,F401
from yolo_multitarea import buscar_modelo_multitarea  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402


class EnvoltorioMovil(nn.Module):
    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        (cajas, puntajes, coeficientes), prototipos = self.modelo(x)
        madurez = torch.softmax(self.modelo.madurez_logits, dim=1)
        return cajas, puntajes, coeficientes, prototipos, madurez


def _inferencia_separada(cabezal, imgsz):
    """Reemplaza Segment._inference: en vez de concatenar cajas, puntajes y
    coeficientes en un solo tensor (lo que arruina la cuantizacion INT8), los
    devuelve por separado y con las cajas normalizadas a 0-1."""
    def inferencia(x):
        forma = x["feats"][0].shape
        if cabezal.dynamic or cabezal.shape != forma:
            from ultralytics.utils.tal import make_anchors
            cabezal.anchors, cabezal.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], cabezal.stride, 0.5))
            cabezal.shape = forma
        cajas = cabezal.decode_bboxes(cabezal.dfl(x["boxes"]), cabezal.anchors.unsqueeze(0)) * (cabezal.strides / imgsz)
        return cajas, x["scores"].sigmoid(), x["mask_coefficient"]
    return inferencia


def preparar(weights, imgsz=800):
    m = buscar_modelo_multitarea(YOLO(weights).model).float().eval()
    m = m.fuse(verbose=False)
    for mod in m.modules():
        if isinstance(mod, Detect):
            mod.export = True
            mod.format = "onnx"
            mod.dynamic = False
            mod._inference = _inferencia_separada(mod, imgsz)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--out", default="modelos_movil")
    ap.add_argument("--nombre", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    nombre = args.nombre or f"palta_multitarea_{args.imgsz}"
    ruta = os.path.join(args.out, nombre + ".onnx")

    modelo = EnvoltorioMovil(preparar(args.weights, args.imgsz))
    x = torch.zeros(1, 3, args.imgsz, args.imgsz)
    with torch.no_grad():
        salidas = modelo(x)
    print("salidas:", [tuple(o.shape) for o in salidas])

    torch.onnx.export(modelo, x, ruta, input_names=["imagen"], output_names=["cajas", "puntajes", "coeficientes", "prototipos", "madurez"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    try:
        import onnx
        import onnxslim
        onnx.save(onnxslim.slim(onnx.load(ruta)), ruta)
    except Exception as e:  # la simplificacion es opcional
        print(f"(sin simplificar: {e})")
    print(f"ONNX guardado: {ruta}  ({os.path.getsize(ruta) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
