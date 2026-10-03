"""
exportar_alternativos.py
Exporta a ONNX los modelos de la comparacion (Etapa 3) para llevarlos a movil
igual que el multitarea (exportar_movil.py): entrada RGB 0-1 con letterbox y
salidas listas para la app, sin pasos extra de normalizacion.

  unet    U-Net ResNet34 multitarea
          entrada "imagen"   float32 [1, 3, 800, 800]  RGB 0-1
          salida  "mascaras" float32 [1, 2, 800, 800]  probabilidad de palta (0) y defecto (1)
          salida  "madurez"  float32 [1, 5]            softmax
  yolo    YOLOv8s-seg sin cabezal de madurez (segmentador de las dos redes)
          mismas salidas que el multitarea menos la madurez:
          "cajas" [1, 4, N] normalizadas 0-1, "puntajes" [1, 2, N], "coeficientes" [1, 32, N], "prototipos"
  resnet  ResNet-34 de madurez (clasificador de las dos redes)
          entrada "imagen" float32 [1, 3, 448, 448] RGB 0-1, salida "madurez" [1, 5] softmax

La normalizacion de ImageNet de la U-Net y la ResNet va dentro del modelo, asi
la app prepara la entrada igual para los tres. Cada ONNX se compara con PyTorch
en una imagen de prueba (diferencia maxima de las salidas).

Uso:
    python scripts/exportar_alternativos.py --unet resultados/comparacion/unet_resnet34/best.pt ^
        --yolo resultados/yolo_multitarea/yolov8s_base_sem1/weights/best.pt ^
        --resnet resultados/comparacion/resnet34_madurez/best.pt
La conversion a TFLite se hace despues con onnx_a_tflite.py.
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clasificador_madurez import crear_modelo  # noqa: E402
from exportar_movil import preparar  # noqa: E402
from unet_multitarea import DESV, MEDIA, UNetMultitarea  # noqa: E402

IMAGEN_PRUEBA = os.path.join("data", "prueba_test")


class Normalizar(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("media", torch.tensor(MEDIA).view(1, 3, 1, 1))
        self.register_buffer("desv", torch.tensor(DESV).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.media) / self.desv


class UNetMovil(nn.Module):
    def __init__(self, modelo):
        super().__init__()
        self.norm, self.modelo = Normalizar(), modelo

    def forward(self, x):
        s = self.modelo(self.norm(x))
        return torch.sigmoid(s["seg"]), torch.softmax(s["madurez"], 1)


class ResNetMovil(nn.Module):
    def __init__(self, modelo):
        super().__init__()
        self.norm, self.modelo = Normalizar(), modelo

    def forward(self, x):
        return torch.softmax(self.modelo(self.norm(x)), 1)


class YoloMovil(nn.Module):
    def __init__(self, modelo):
        super().__init__()
        self.modelo = modelo

    def forward(self, x):
        (cajas, puntajes, coeficientes), prototipos = self.modelo(x)
        return cajas, puntajes, coeficientes, prototipos


def cargar_estado(ruta):
    ck = torch.load(ruta, map_location="cpu", weights_only=False)
    return ck["modelo"] if isinstance(ck, dict) and "modelo" in ck else ck


def entrada_prueba(lado):
    """Una foto real del conjunto de prueba (o ruido si no hay) en RGB 0-1."""
    fotos = sorted(os.listdir(IMAGEN_PRUEBA)) if os.path.isdir(IMAGEN_PRUEBA) else []
    if fotos:
        img = cv2.cvtColor(cv2.imread(os.path.join(IMAGEN_PRUEBA, fotos[0])), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (lado, lado), interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)
    return torch.rand(1, 3, lado, lado)


def exportar(modelo, lado, ruta, entradas, salidas):
    modelo = modelo.float().eval()
    x = entrada_prueba(lado)
    with torch.no_grad():
        ref = [o.numpy() for o in modelo(x)]
    torch.onnx.export(modelo, torch.zeros(1, 3, lado, lado), ruta, input_names=entradas, output_names=salidas,
                      opset_version=17, do_constant_folding=True, dynamo=False)
    try:
        import onnx
        import onnxslim
        onnx.save(onnxslim.slim(onnx.load(ruta)), ruta)
    except Exception as e:  # la simplificacion es opcional
        print(f"  (sin simplificar: {e})")
    import onnxruntime as ort
    s = ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])
    obt = s.run(None, {s.get_inputs()[0].name: x.numpy()})
    dif = max(float(np.abs(a - b).max()) for a, b in zip(ref, obt))
    formas = ", ".join(f"{n} {tuple(o.shape)}" for n, o in zip(salidas, obt))
    print(f"  {ruta}: {os.path.getsize(ruta) / 1e6:.1f} MB | {formas} | dif. max. con PyTorch {dif:.2e}")
    if dif > 1e-3:
        print("  ATENCION: el ONNX no reproduce a PyTorch")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", default="")
    ap.add_argument("--yolo", default="")
    ap.add_argument("--resnet", default="")
    ap.add_argument("--out", default="modelos_movil")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.unet:
        m = UNetMultitarea(preentrenado=False)
        m.load_state_dict(cargar_estado(args.unet))
        exportar(UNetMovil(m), 800, os.path.join(args.out, "unet_resnet34_800.onnx"), ["imagen"], ["mascaras", "madurez"])
    if args.yolo:
        exportar(YoloMovil(preparar(args.yolo, 800, multitarea=False)), 800, os.path.join(args.out, "yolov8s_seg_800.onnx"),
                 ["imagen"], ["cajas", "puntajes", "coeficientes", "prototipos"])
    if args.resnet:
        ck = torch.load(args.resnet, map_location="cpu", weights_only=False)
        lado = ck.get("config", {}).get("imgsz", 448) if isinstance(ck, dict) else 448
        m = crear_modelo(False)
        m.load_state_dict(cargar_estado(args.resnet))
        exportar(ResNetMovil(m), lado, os.path.join(args.out, f"resnet34_madurez_{lado}.onnx"), ["imagen"], ["madurez"])


if __name__ == "__main__":
    main()
