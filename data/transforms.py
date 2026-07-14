"""
data/transforms.py

数据增强（Albumentations）
Resize → RandomCrop → 几何/颜色增强 → Normalize
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size: int = 784):
    resize_size = int(image_size * 1.15)

    return A.Compose([
        A.Resize(resize_size, resize_size),
        A.RandomCrop(image_size, image_size),

        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.90, 1.10),
            rotate=(-15, 15),
            border_mode=0,
            p=0.5
        ),

        A.ElasticTransform(alpha=1, sigma=20, border_mode=0, p=0.2),

        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),

        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_val_transform(image_size: int = 784):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def get_test_transform(image_size: int = 784):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
