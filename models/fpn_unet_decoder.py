"""
FPN + UNet Hybrid Decoder（适配 ViT 扁平特征）

ViT 所有层输出相同空间分辨率（image_size/patch_size = 784/14 = 56×56）。
解码策略：同分辨率融合 → 逐级2×上采样 → 最后interpolate到精确尺寸。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FPNBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return self.conv(x)


class FPN_UNet(nn.Module):
    def __init__(self, num_classes, feat_dim=768, fpn_dim=256):
        super().__init__()

        # FPN projection：3层ViT特征 → 统一256通道
        self.p4 = FPNBlock(feat_dim, fpn_dim)
        self.p8 = FPNBlock(feat_dim, fpn_dim)
        self.p16 = FPNBlock(feat_dim, fpn_dim)

        # 融合后降维
        self.fusion = ConvBlock(fpn_dim, fpn_dim // 2)   # 256→128

        # 逐级2×上采样：56→112→224→448→896
        self.up1 = ConvBlock(128, 64)
        self.up2 = ConvBlock(64, 32)
        self.up3 = ConvBlock(32, 32)
        self.up4 = ConvBlock(32, 32)

        self.head = nn.Conv2d(32, num_classes, 1)

    def forward(self, feats, output_size=None):
        """
        feats: {"f4","f8","f16"} — 全部同分辨率 [B, 768, 56, 56]
        output_size: (H,W) 输出尺寸，默认=input_size
        """
        f4, f8, f16 = feats["f4"], feats["f8"], feats["f16"]

        # 1×1 投影
        p16 = self.p16(f16)   # [B, 256, 56, 56]
        p8  = self.p8(f8)
        p4  = self.p4(f4)

        # 同分辨率 add 融合
        x = p16 + p8 + p4     # [B, 256, 56, 56]

        x = self.fusion(x)    # [B, 128, 56, 56]

        # 逐级 2× 上采样 + 精炼
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up1(x)       # [B, 64, 112, 112]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x)       # [B, 32, 224, 224]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x)       # [B, 32, 448, 448]

        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up4(x)       # [B, 32, 896, 896]

        # 精确resize到目标尺寸
        if output_size is not None:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)

        return self.head(x)
