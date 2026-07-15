"""
losses/loss.py

总损失：BCE(pos_weight) + Dice + Boundary + Connectivity
v4：新增可微分形态学闭运算损失，增强边缘连续性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg


class DiceLoss(nn.Module):
    """Smooth Dice Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(
            target.squeeze(1).long(), prob.shape[1]
        ).permute(0, 3, 1, 2).float()

        inter = (prob * target_oh).sum()
        union = prob.sum() + target_oh.sum()

        return 1 - (2 * inter + self.smooth) / (union + self.smooth)


class BoundaryLoss(nn.Module):
    """Laplacian边缘一致性损失"""
    def __init__(self):
        super().__init__()
        self.laplacian = torch.tensor(
            [[[[-1., -1., -1.],
               [-1.,  8., -1.],
               [-1., -1., -1.]]]], dtype=torch.float32
        )

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)[:, 1:2]

        k = self.laplacian.to(prob.device)
        edge_pred = F.conv2d(prob, k, padding=1).abs()

        gt = target.float()
        ones = torch.ones(1, 1, 3, 3, device=gt.device)
        edge_gt = F.conv2d(gt, ones, padding=1)
        edge_gt = ((edge_gt > 0) & (edge_gt < 9)).float()

        return F.l1_loss(edge_pred, edge_gt)


class ConnectivityLoss(nn.Module):
    """
    可微分形态学闭运算损失
    对预测概率图做 soft-close（dilate→erode），
    鼓励断裂处（小间隙）被填补，增强边缘拓扑连续性。

    原理：
      close(x) = erode(dilate(x))
      dilate ≈ max_pool, erode ≈ -max_pool(-x)
      kernel_size=7 → 填充 ≤6px 的间隙
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

    def soft_close(self, x):
        # dilation
        d = F.max_pool2d(x, self.kernel_size, stride=1, padding=self.pad)
        # erosion
        e = -F.max_pool2d(-d, self.kernel_size, stride=1, padding=self.pad)
        return e

    def forward(self, logits, target):
        prob = torch.softmax(logits, dim=1)[:, 1:2]

        pred_closed = self.soft_close(prob)
        gt_closed = self.soft_close(target.float())

        return F.l1_loss(pred_closed, gt_closed)


class TotalLoss(nn.Module):
    """
    组合损失 = bce_weight*BCE + dice_weight*Dice
             + boundary_weight*Boundary + connectivity_weight*Connectivity
    """
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
        self.connectivity = ConnectivityLoss(kernel_size=7)

        self.pos_weight = torch.tensor([cfg.bce_pos_weight])

    def forward(self, logits, target):
        # BCE
        pw = self.pos_weight.to(logits.device)
        bce = F.binary_cross_entropy_with_logits(
            logits[:, 1:2], target.float(), pos_weight=pw
        )

        # Dice
        dice = self.dice(logits, target)

        # Boundary (edge)
        bd = self.boundary(logits, target)

        # Connectivity (topology)
        conn = self.connectivity(logits, target)

        return (cfg.bce_weight * bce
                + cfg.dice_weight * dice
                + cfg.boundary_weight * bd
                + cfg.connectivity_weight * conn)
