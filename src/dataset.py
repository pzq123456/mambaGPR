# src/dataset.py
import torch
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class GPRDataset(Dataset):
    def __init__(self, img_dir, ann_dir=None, transform=None, is_pretext=False,
                 mean=None, std=None):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir) if ann_dir else None
        self.img_files = sorted(list(self.img_dir.glob("*.png")))
        self.transform = transform
        self.is_pretext = is_pretext

        # 允许从外部注入归一化参数（如果没有则使用默认值）
        self.mean = mean if mean is not None else [0.499, 0.499, 0.499]
        self.std = std if std is not None else [0.085, 0.085, 0.085]

        if not self.is_pretext and self.ann_dir is None:
            raise ValueError("分割模式下必须提供 ann_dir 路径。")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")

        if self.is_pretext:
            if self.transform:
                img = self.transform(img)
            else:
                img = F.to_tensor(img)
                img = F.normalize(img, mean=self.mean, std=self.std)
            return img

        # 分割模式
        ann_path = self.ann_dir / img_path.name
        if not ann_path.exists():
            raise FileNotFoundError(f"未找到对应的标注文件: {ann_path}")
        mask = Image.open(ann_path).convert("L")
        img_tensor, mask_tensor = self._sync_transform(img, mask)
        return img_tensor, mask_tensor

    def _sync_transform(self, img_pil, mask_pil):
        # 几何增强
        if random.random() > 0.5:
            img_pil = F.hflip(img_pil)
            mask_pil = F.hflip(mask_pil)
        # 光度增强
        if random.random() > 0.5:
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            img_pil = F.adjust_brightness(img_pil, brightness)
            img_pil = F.adjust_contrast(img_pil, contrast)

        img_tensor = F.to_tensor(img_pil)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        return img_tensor, mask_tensor