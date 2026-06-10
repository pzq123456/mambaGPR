# src/dataset.py
import torch
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class GPRDataset(Dataset):
    def __init__(self, img_dir, ann_dir, split="train", mean=None, std=None):
        """
        工业落地版有监督图像分割 Dataset
        """
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        self.split = split.lower()

        # 获取所有图片文件
        self.img_files = sorted(list(self.img_dir.glob("*.png")))
        if len(self.img_files) == 0:
            raise FileNotFoundError(f"❌ 在路径 {img_dir} 下未找到任何 .png 格式雷达图")
        
        self.mean = mean if mean is not None else [0.499, 0.499, 0.499]
        self.std = std if std is not None else [0.085, 0.085, 0.085]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_dir / img_path.name
        
        if not ann_path.exists():
            raise FileNotFoundError(f"❌ 找不到与图像对应的标注掩码文件: {ann_path}")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")
        
        # 工业落地双保险：强制尺寸严格对其
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        img_tensor, mask_tensor = self._sync_transform(img, mask)
        return img_tensor, mask_tensor

    def _sync_transform(self, img_pil, mask_pil):
        # 仅在训练阶段应用稳健的工程化数据增强
        if self.split == "train":
            if random.random() > 0.5:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)
            if random.random() > 0.3:
                img_pil = F.adjust_brightness(img_pil, random.uniform(0.85, 1.15))
                img_pil = F.adjust_contrast(img_pil, random.uniform(0.85, 1.15))

        # 转换为 Tensor 归一化
        img_tensor = F.to_tensor(img_pil)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
        
        # 处理掩码标签
        mask_np = np.array(mask_pil)
        if mask_np.max() > 1:
            mask_np = (mask_np > 128).astype(np.int64)  # 容错二值化阈值
            
        mask_tensor = torch.from_numpy(mask_np).long()
        return img_tensor, mask_tensor