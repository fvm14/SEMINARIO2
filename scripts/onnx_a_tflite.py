"""
onnx_a_tflite.py
Convierte un ONNX a TFLite como en la Etapa 2: onnx2tf pasa el grafo de NCHW a
NHWC con el convertidor de TensorFlow y genera

  <nombre>_float32.tflite              referencia sin cuantizar
  <nombre>_float16.tflite              pesos en media precision
  <nombre>_dynamic_range_quant.tflite  pesos INT8, activaciones en float (no requiere calibracion)

Se ejecuta con un entorno que tenga tensorflow y onnx2tf (aparte del de
PyTorch, para no mezclar dependencias):
    python scripts/onnx_a_tflite.py modelos_movil/unet_resnet34_800.onnx
"""

import argparse
import glob
import os
import shutil
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx", nargs="+")
    ap.add_argument("--out", default="modelos_movil")
    args = ap.parse_args()

    import onnx2tf

    os.makedirs(args.out, exist_ok=True)
    for ruta in args.onnx:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        tmp = tempfile.mkdtemp(prefix="onnx2tf_")
        try:
            onnx2tf.convert(input_onnx_file_path=ruta, output_folder_path=tmp, non_verbose=True,
                            tflite_backend="tf_converter", output_dynamic_range_quantized_tflite=True)
            for sufijo in ("float32", "float16", "dynamic_range_quant"):
                origen = glob.glob(os.path.join(tmp, f"*_{sufijo}.tflite"))
                if not origen:
                    print(f"  falta la variante {sufijo}")
                    continue
                destino = os.path.join(args.out, f"{nombre}_{sufijo}.tflite")
                shutil.copy(origen[0], destino)
                print(f"{destino}: {os.path.getsize(destino) / 1e6:.1f} MB")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
