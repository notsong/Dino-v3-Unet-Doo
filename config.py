"""
config.py
v4 — DINOv3 + 连接性增强 + 两阶段训练 + 日志系统
"""
from dataclasses import dataclass
import os


@dataclass
class Config:

    # ===========================
    # Dataset
    # ===========================
    dataset_root: str = "./dataset"

    train_image_dir: str = os.path.join(dataset_root, "train/images")
    train_mask_dir: str = os.path.join(dataset_root, "train/masks")

    val_image_dir: str = os.path.join(dataset_root, "val/images")
    val_mask_dir: str = os.path.join(dataset_root, "val/masks")

    test_image_dir: str = os.path.join(dataset_root, "test/images")

    unlabeled_dir: str = os.path.join(dataset_root, "unlabeled")

    # ===========================
    # Model — DINOv3
    # ===========================
    model_type: str = "dinov3_fpn_unet_v4"

    backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"

    num_classes: int = 2

    in_channels: int = 3

    image_size: int = 784  # 49*16, DINOv3 ViT-B/16 patch=16

    use_multi_layer_features: bool = True

    fpn_channels: int = 256

    decoder_type: str = "fpn_unet"

    # ===========================
    # Two-Stage Training
    # ===========================
    epochs: int = 100

    # 阶段1：冻结backbone，只训练decoder
    freeze_backbone: bool = True
    freeze_epochs: int = 60

    # 阶段2：解冻backbone，全模型微调（更低LR）
    unfreeze_lr: float = 1e-5

    batch_size: int = 4

    num_workers: int = 4

    learning_rate: float = 1e-4

    weight_decay: float = 1e-4

    grad_clip: float = 1.0

    amp: bool = True

    seed: int = 42

    device: str = "cuda"

    # ===========================
    # Scheduler
    # ===========================
    warmup_epochs: int = 5

    min_lr: float = 1e-6

    # ===========================
    # Loss — 强化边缘召回
    # ===========================
    dice_weight: float = 0.45

    bce_weight: float = 0.20

    boundary_weight: float = 0.2

    # 可微分形态学闭运算损失，鼓励断裂处自动连接
    connectivity_weight: float = 0.15

    # BCE正样本权重：漏判边缘的惩罚倍率
    bce_pos_weight: float = 12.0

    # ===========================
    # Validation & Logging
    # ===========================
    val_interval: int = 1

    log_dir: str = "./logs"

    # ===========================
    # Checkpoint
    # ===========================
    save_dir: str = "./checkpoints"

    resume: str = ""

    pretrained: bool = True

    save_best_dice: bool = True
    save_best_iou: bool = True

    # ===========================
    # Sliding Window Inference
    # ===========================
    crop_size: int = 784

    stride: int = 512

    # 推理阈值（可调低增强连续性）
    infer_threshold: float = 0.3

    # 推理后处理：形态学闭运算核大小（0=禁用）
    infer_close_kernel: int = 3

    # ===========================
    # ONNX Export
    # ===========================
    onnx_path: str = "./onnx/model.onnx"

    # ===========================
    # Output
    # ===========================
    output_dir: str = "./output"

    visualization_dir: str = "./output/vis"


cfg = Config()
