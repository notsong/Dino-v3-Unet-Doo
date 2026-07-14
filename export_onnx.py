"""
export_onnx.py

DINOv3 + FPN-UNet  ONNX导出（GPU版，FP16）
"""
import os
import torch

from config import cfg
from models.dinov3_segmentation import DINOv3Seg


def export_onnx(weight_path, onnx_path, device):
    print(f"Loading model from {weight_path} ...")
    model = DINOv3Seg(cfg).half().to(device)
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    print(f"  epoch={ckpt['epoch']}, IoU={ckpt.get('iou',0):.4f}, Dice={ckpt.get('dice',0):.4f}")

    dummy_input = torch.randn(1, 3, cfg.image_size, cfg.image_size,
                              dtype=torch.float16, device=device)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
    )

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX exported & verified: {onnx_path}")
    print(f"  Size: {os.path.getsize(onnx_path)/1024/1024:.1f} MB")


if __name__ == "__main__":
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    export_onnx(
        weight_path=os.path.join(cfg.save_dir, "best_dice.pth"),
        onnx_path=cfg.onnx_path,
        device=device
    )
