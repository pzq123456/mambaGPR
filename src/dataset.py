# src\dataset.py
import torch
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

class GPRDataset(Dataset):
    """
    针对 GPR 任务优化的数据加载类。
    
    特点：
    - 路径管理：使用 pathlib 兼容跨平台。
    - 物理增强：仅允许水平翻转，防止破坏地层深度顺序。
    - 信号模拟：通过亮度与对比度波动模拟雷达增益（Gain）变化。
    - 模式切换：支持有监督分割（Downstream）与无监督重构（Pretext）。
    """
    def __init__(self, img_dir, ann_dir=None, transform=None, is_pretext=False):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir) if ann_dir else None
        
        # 严格过滤，确保只读取 png 文件并排序
        self.img_files = sorted(list(self.img_dir.glob("*.png")))
        self.transform = transform
        self.is_pretext = is_pretext
        
        # 归一化参数：基于之前的扫描统计结果
        # 均值约 0.5，标准差约 0.085
        self.mean = [0.499, 0.499, 0.499]
        self.std = [0.085, 0.085, 0.085]

        if not self.is_pretext and self.ann_dir is None:
            raise ValueError("分割模式下必须提供 ann_dir 路径。")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # --- 1. 预训练/自监督模式 (Pretext) ---
        if self.is_pretext:
            if self.transform:
                img = self.transform(img)
            else:
                img = F.to_tensor(img)
                img = F.normalize(img, mean=self.mean, std=self.std)
            return img 

        # --- 2. 分割模式 (Supervised Segmentation) ---
        ann_path = self.ann_dir / img_path.name
        if not ann_path.exists():
            raise FileNotFoundError(f"未找到对应的标注文件: {ann_path}")
            
        mask = Image.open(ann_path).convert("L")
        
        # 执行同步增强逻辑
        img_tensor, mask_tensor = self._sync_transform(img, mask)
        
        return img_tensor, mask_tensor

    def _sync_transform(self, img_pil, mask_pil):
        """
        核心同步增强：保持图像与掩码的几何一致性。
        """
        # A. 几何增强 (Geometric)
        # 模拟雷达反向行驶：水平翻转是安全的
        if random.random() > 0.5:
            img_pil = F.hflip(img_pil)
            mask_pil = F.hflip(mask_pil)
            
        # B. 光度增强 (Photometric)
        # 模拟不同的雷达增益或土壤介电常数引起的信号强度波动
        if random.random() > 0.5:
            # 波动范围控制在 10% 以内，防止信号失真
            brightness = random.uniform(0.9, 1.1)
            contrast = random.uniform(0.9, 1.1)
            img_pil = F.adjust_brightness(img_pil, brightness)
            img_pil = F.adjust_contrast(img_pil, contrast)

        # C. 转换为 Tensor
        # ToTensor 会将 [0, 255] 缩放到 [0.0, 1.0]
        img_tensor = F.to_tensor(img_pil)  # type: ignore
        
        # D. 标准化
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)

        # E. Mask 处理
        # 直接使用 numpy 转换，确保类别标签 (0, 1) 不会被插值改变
        mask_tensor = torch.from_numpy(np.array(mask_pil)).long()
        
        return img_tensor, mask_tensor