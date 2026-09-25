"""
transforms.py
Augmentations para el dataset multitarea (segmentacion palta+defecto + madurez).

Deliberadamente SIN mosaic ni mixup: esas tecnicas mezclan pixeles de varias
imagenes, lo cual corromperia la etiqueta de madurez (una imagen resultante
podria tener pixeles de madurez 1 y madurez 4 mezclados, pero solo puede
tener UNA etiqueta de madurez). El pipeline original de YOLO si las usaba
porque madurez nunca fue un target de entrenamiento; aqui si lo es.
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 512):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, fill=255,
                      fill_mask=0, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25,
                                  val_shift_limit=20, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(img_size: int = 512):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )
