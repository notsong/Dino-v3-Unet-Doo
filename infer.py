"""
infer.py

DINOv3 + FPN-UNet 极速推理
- FP16 推理
- 批量滑窗（单次前向处理所有窗口）
- 大图自动降采样减少窗口数
- 形态学后处理增强连续性
"""
import os
import cv2
import numpy as np
import torch

from config import cfg
from models.dinov3_segmentation import DINOv3Seg


# FP16 预计算常量
MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float16).view(1, 3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float16).view(1, 3, 1, 1)


def load_model(weight_path, device):
    model = DINOv3Seg(cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"Loaded: {weight_path} (epoch={ckpt.get('epoch','?')}, "
          f"IoU={ckpt.get('iou',0):.4f}, Dice={ckpt.get('dice',0):.4f})")
    return model


def enhance_connectivity(mask, kernel_size=3):
    if kernel_size <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def single_infer(model, image, device):
    """单张图直推（≤784 的小图）"""
    h, w = image.shape[:2]
    inp = cv2.resize(image, (cfg.image_size, cfg.image_size)).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).half().to(device)
    inp = (inp - MEAN.to(device)) / STD.to(device)
    with torch.no_grad():
        prob = torch.softmax(model(inp), dim=1)[0, 1].float().cpu().numpy()
    if prob.shape != (h, w):
        prob = cv2.resize(prob, (w, h))
    return prob


def _make_blend_weight(crop):
    """2D余弦窗：中心权重=1，边缘→0，消除滑窗边界伪影"""
    wy = np.hanning(crop).astype(np.float32)
    wx = np.hanning(crop).astype(np.float32)
    return np.outer(wy, wx)  # [crop, crop]


def batched_sliding(model, image, device, crop=784, stride=520):
    """批量滑窗 + 余弦加权 + 反射填充，消除边缘伪影"""
    h, w = image.shape[:2]

    # 反射填充：确保边缘像素也被多个窗口重叠覆盖
    pad = stride
    padded = cv2.copyMakeBorder(image, 0, pad, 0, pad, cv2.BORDER_REFLECT)
    ph, pw = padded.shape[:2]

    xs = sorted(set(min(x, pw - crop) for x in range(0, pw, stride)))
    ys = sorted(set(min(y, ph - crop) for y in range(0, ph, stride)))
    weight = _make_blend_weight(crop)

    patches, positions = [], []
    for y1 in ys:
        for x1 in xs:
            p = cv2.resize(padded[y1:y1+crop, x1:x1+crop], (crop, crop))
            patches.append(p.astype(np.float32) / 255.0)
            positions.append((y1, x1))

    batch = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).half().to(device)
    batch = (batch - MEAN.to(device)) / STD.to(device)

    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)[:, 1].float().cpu().numpy()

    # 只在原图区域累加，但利用重叠覆盖消除边缘效应
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)
    for (y1, x1), prob in zip(positions, probs):
        # 裁剪到原图范围内
        yo1, xo1 = max(0, y1), max(0, x1)
        yo2, xo2 = min(h, y1+crop), min(w, x1+crop)
        wy1, wx1 = yo1 - y1, xo1 - x1
        wy2, wx2 = yo2 - y1, xo2 - x1
        prob_map[yo1:yo2, xo1:xo2] += prob[wy1:wy2, wx1:wx2] * weight[wy1:wy2, wx1:wx2]
        count_map[yo1:yo2, xo1:xo2] += weight[wy1:wy2, wx1:wx2]

    return prob_map / (count_map + 1e-6)


SCALE = 0.85  # 大图缩放比例（速度/质量平衡）

def infer_single(model, image_path, device):
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # 大图：缩放到 0.85x 再滑窗（窗口数减半，质量几乎无损）
    if max(h, w) > 1200:
        nh, nw = int(h * SCALE), int(w * SCALE)
        small = cv2.resize(image, (nw, nh))
        if max(nh, nw) <= cfg.image_size:
            prob = single_infer(model, small, device)
        else:
            prob = batched_sliding(model, small, device)
        prob = cv2.resize(prob, (w, h))
    elif max(h, w) > cfg.image_size * 1.2:
        prob = batched_sliding(model, image, device)
    else:
        prob = single_infer(model, image, device)

    mask = (prob > cfg.infer_threshold).astype(np.uint8)
    mask = enhance_connectivity(mask, cfg.infer_close_kernel)
    return mask, prob


def save_result(mask, prob, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path + "_mask.png", mask * 255)
    cv2.imwrite(save_path + "_prob.png", (prob * 255).astype(np.uint8))


if __name__ == "__main__":
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(cfg.save_dir, "best_dice.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.save_dir, "best_iou.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(cfg.save_dir, "last.pth")

    model = load_model(ckpt_path, device)
    _ = MEAN.to(device); _ = STD.to(device)  # pin to GPU

    test_dir = cfg.test_image_dir
    if not os.path.exists(test_dir):
        print(f"Test dir not found: {test_dir}")
        exit(1)

    print(f"Threshold: {cfg.infer_threshold}, Close: {cfg.infer_close_kernel}")

    for name in sorted(os.listdir(test_dir)):
        if not name.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            continue
        path = os.path.join(test_dir, name)
        print(f"Infer: {name}")
        mask, prob = infer_single(model, path, device)
        save_result(mask, prob,
                    os.path.join(cfg.output_dir, "infer", os.path.splitext(name)[0]))

    print("Done.")
