"""
DINOv3 + FPN-UNet 完整模型
"""

import torch.nn as nn
from models.dinov3_encoder import DINOv3Encoder
from models.fpn_unet_decoder import FPN_UNet


class DINOv3Seg(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = DINOv3Encoder(
            cfg.backbone_name,
            trainable=cfg.freeze_backbone == False
        )

        self.decoder = FPN_UNet(cfg.num_classes)

    def forward(self, x):
        feats = self.encoder(x)
        # ViT特征全部分辨率相同，传目标尺寸让decoder对齐
        out = self.decoder(feats, output_size=x.shape[2:])
        return out