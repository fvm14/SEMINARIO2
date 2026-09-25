"""
model.py
Modelo multitarea: backbone ResNet34 (preentrenado en ImageNet) compartido
por dos cabezales:
  1) Decoder tipo FPN -> 2 mascaras binarias independientes (palta, defecto)
  2) Cabezal de clasificacion -> 5 clases de madurez

Las dos mascaras NO son mutuamente excluyentes (el defecto esta dentro del
area de la palta), por eso el cabezal de segmentacion usa sigmoid por canal
en vez de softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Encoder(nn.Module):
    """Extrae mapas de features en 4 resoluciones (stride 4, 8, 16, 32)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )  # stride 4
        self.layer1 = backbone.layer1  # stride 4,  64 canales
        self.layer2 = backbone.layer2  # stride 8,  128 canales
        self.layer3 = backbone.layer3  # stride 16, 256 canales
        self.layer4 = backbone.layer4  # stride 32, 512 canales

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c1, c2, c3, c4


class FPNDecoder(nn.Module):
    """Fusiona c1..c4 top-down (estilo FPN) y produce logits de segmentacion
    a la resolucion de entrada."""

    def __init__(self, canales_in=(64, 128, 256, 512), canales_fpn=128, n_clases_seg=2):
        super().__init__()
        c1_ch, c2_ch, c3_ch, c4_ch = canales_in

        # convs laterales 1x1 para unificar todos los niveles a canales_fpn
        self.lat1 = nn.Conv2d(c1_ch, canales_fpn, kernel_size=1)
        self.lat2 = nn.Conv2d(c2_ch, canales_fpn, kernel_size=1)
        self.lat3 = nn.Conv2d(c3_ch, canales_fpn, kernel_size=1)
        self.lat4 = nn.Conv2d(c4_ch, canales_fpn, kernel_size=1)

        # smoothing 3x3 tras cada suma top-down
        self.smooth1 = nn.Conv2d(canales_fpn, canales_fpn, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(canales_fpn, canales_fpn, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(canales_fpn, canales_fpn, kernel_size=3, padding=1)

        self.cabezal_seg = nn.Sequential(
            nn.Conv2d(canales_fpn, canales_fpn, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_fpn),
            nn.ReLU(inplace=True),
            nn.Conv2d(canales_fpn, n_clases_seg, kernel_size=1),
        )

    @staticmethod
    def _upsample_add(x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False) + y

    def forward(self, c1, c2, c3, c4, tamano_salida):
        p4 = self.lat4(c4)
        p3 = self.smooth3(self._upsample_add(p4, self.lat3(c3)))
        p2 = self.smooth2(self._upsample_add(p3, self.lat2(c2)))
        p1 = self.smooth1(self._upsample_add(p2, self.lat1(c1)))

        logits = self.cabezal_seg(p1)  # stride 4 respecto a la entrada
        logits = F.interpolate(logits, size=tamano_salida, mode="bilinear", align_corners=False)
        return logits  # (B, 2, H, W) -- canal 0 = palta, canal 1 = defecto


class CabezalMadurez(nn.Module):
    def __init__(self, canales_in=512, n_clases=5, dropout=0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(canales_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_clases),
        )

    def forward(self, c4):
        return self.mlp(self.pool(c4))  # (B, n_clases) logits


class PaltaMultitaskModel(nn.Module):
    def __init__(self, pretrained: bool = True, n_clases_madurez: int = 5):
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=pretrained)
        self.decoder_seg = FPNDecoder(n_clases_seg=2)
        self.cabezal_madurez = CabezalMadurez(n_clases=n_clases_madurez)

    def forward(self, x):
        tamano_entrada = x.shape[-2:]
        c1, c2, c3, c4 = self.encoder(x)
        logits_seg = self.decoder_seg(c1, c2, c3, c4, tamano_entrada)
        logits_madurez = self.cabezal_madurez(c4)
        return {"seg": logits_seg, "madurez": logits_madurez}


if __name__ == "__main__":
    # smoke test manual: python model.py
    modelo = PaltaMultitaskModel(pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    salida = modelo(x)
    print("seg:", salida["seg"].shape)         # esperado: (2, 2, 512, 512)
    print("madurez:", salida["madurez"].shape)  # esperado: (2, 5)
    n_params = sum(p.numel() for p in modelo.parameters())
    print(f"parametros totales: {n_params:,}")
