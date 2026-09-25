"""
dataset.py
Dataset de PyTorch para el modelo multitarea (segmentacion palta+defecto
y clasificacion de madurez 1-5).

Lee data/splits/split_manifest.csv (generado por split_dataset.py) y carga
las imagenes/labels_seg directamente desde data/raw/ (no hay copia fisica
por split, el manifest filtra).
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # modulos compartidos en scripts/


import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from rasterize_utils import poligonos_a_mascaras


class PaltaMultitaskDataset(Dataset):
    def __init__(self, manifest_csv: str, raw_dir: str, split: str,
                 transform=None):
        """
        manifest_csv: ruta a data/splits/split_manifest.csv
        raw_dir: ruta a data/raw (contiene images/ y labels_seg/)
        split: "train" | "val" | "test"
        transform: albumentations.Compose (ver transforms.py); si es None,
                   solo convierte a tensor sin augmentations ni resize.
        """
        manifest = pd.read_csv(manifest_csv)
        self.filas = manifest[manifest["split"] == split].reset_index(drop=True)
        if len(self.filas) == 0:
            raise ValueError(f"El split '{split}' no tiene filas en {manifest_csv}")

        self.images_dir = os.path.join(raw_dir, "images")
        self.labels_seg_dir = os.path.join(raw_dir, "labels_seg")
        self.transform = transform

    def __len__(self):
        return len(self.filas)

    def __getitem__(self, idx):
        fila = self.filas.iloc[idx]
        archivo = fila["archivo"]
        stem = archivo.rsplit(".", 1)[0]
        madurez = int(fila["madurez"])  # 1..5

        img_path = os.path.join(self.images_dir, archivo)
        txt_path = os.path.join(self.labels_seg_dir, stem + ".txt")

        imagen = cv2.imread(img_path)
        if imagen is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
        alto, ancho = imagen.shape[:2]

        mascara_palta, mascara_defecto = poligonos_a_mascaras(txt_path, alto, ancho)

        if self.transform is not None:
            aug = self.transform(image=imagen, masks=[mascara_palta, mascara_defecto])
            imagen_t = aug["image"]  # ya es tensor CHW normalizado (via ToTensorV2)
            m_palta_t = torch.as_tensor(aug["masks"][0], dtype=torch.float32)
            m_defecto_t = torch.as_tensor(aug["masks"][1], dtype=torch.float32)
        else:
            imagen_t = torch.from_numpy(imagen.transpose(2, 0, 1)).float() / 255.0
            m_palta_t = torch.from_numpy(mascara_palta).float()
            m_defecto_t = torch.from_numpy(mascara_defecto).float()

        mascaras_t = torch.stack([m_palta_t, m_defecto_t], dim=0)  # (2, H, W)
        madurez_t = torch.tensor(madurez - 1, dtype=torch.long)  # 0..4 para CrossEntropyLoss

        return {
            "imagen": imagen_t,
            "mascaras": mascaras_t,
            "madurez": madurez_t,
            "archivo": archivo,
        }


if __name__ == "__main__":
    # smoke test manual: python dataset.py
    from transforms import get_train_transforms

    ds = PaltaMultitaskDataset(
        manifest_csv=os.path.join("data", "splits", "split_manifest.csv"),
        raw_dir=os.path.join("data", "raw"),
        split="train",
        transform=get_train_transforms(img_size=512),
    )
    print(f"Dataset train: {len(ds)} muestras")
    muestra = ds[0]
    print("imagen:", muestra["imagen"].shape, muestra["imagen"].dtype)
    print("mascaras:", muestra["mascaras"].shape, muestra["mascaras"].dtype,
          "min/max:", muestra["mascaras"].min().item(), muestra["mascaras"].max().item())
    print("madurez:", muestra["madurez"].item(), "archivo:", muestra["archivo"])
