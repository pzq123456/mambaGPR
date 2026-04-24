import torch
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as F

class GPRDataset(Dataset):
    def __init__(self, img_dir, ann_dir=None, split="train", is_pretext=False,
                 mean=None, std=None, align_size=8):
        """
        Args:
            img_dir: 图像文件夹路径
            ann_dir: 标注文件夹路径
            split: "train", "val", 或 "test"
            is_pretext: 是否为自监督预训练模式
            mean: 归一化均值
            std: 归一化标准差
            align_size: 强制对齐的数据倍数 (默认为 8)
        """
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir) if ann_dir else None
        self.split = split.lower()
        self.is_pretext = is_pretext
        self.align_size = align_size

        # 获取所有图片文件
        self.img_files = sorted(list(self.img_dir.glob("*.png")))
        self.real_len = len(self.img_files)
        
        if self.real_len == 0:
            raise FileNotFoundError(f"在 {img_dir} 下未找到 .png 文件")

        # --- 核心修改：计算补齐后的长度 ---
        # 补齐逻辑：如果数据量不是 align_size 的倍数，则向上取整
        self.padded_len = ((self.real_len + self.align_size - 1) // self.align_size) * self.align_size
        
        # 归一化参数
        self.mean = mean if mean is not None else [0.499, 0.499, 0.499]
        self.std = std if std is not None else [0.085, 0.085, 0.085]

        if not self.is_pretext and self.ann_dir is None:
            raise ValueError(f"当前模式为 {self.split} 且非预训练模式，必须提供 ann_dir 路径。")

    def __len__(self):
        # 返回补齐后的长度，确保 DataLoader 拿到的总数是 align_size 的倍数
        return self.padded_len

    def __getitem__(self, idx):
        # 使用取模操作，当 idx 超过 real_len 时，会重新从头开始取数据
        actual_idx = idx % self.real_len
        
        img_path = self.img_files[actual_idx]
        img = Image.open(img_path).convert("RGB")

        # --- 模式 1: Pretext (自监督) ---
        if self.is_pretext:
            img_tensor = F.to_tensor(img)
            img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
            return img_tensor

        # --- 模式 2: Downstream (有监督分割) ---
        ann_path = self.ann_dir / img_path.name
        mask = Image.open(ann_path).convert("L")
        
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        img_tensor, mask_tensor = self._sync_transform(img, mask)
        return img_tensor, mask_tensor

    def _sync_transform(self, img_pil, mask_pil):
        if self.split == "train":
            if random.random() > 0.5:
                img_pil = F.hflip(img_pil)
                mask_pil = F.hflip(mask_pil)
            if random.random() > 0.3:
                img_pil = F.adjust_brightness(img_pil, random.uniform(0.8, 1.2))
                img_pil = F.adjust_contrast(img_pil, random.uniform(0.8, 1.2))

        img_tensor = F.to_tensor(img_pil)
        img_tensor = F.normalize(img_tensor, mean=self.mean, std=self.std)
        
        mask_np = np.array(mask_pil)
        if mask_np.max() > 1:
            mask_np = (mask_np > 128).astype(np.int64)
            
        mask_tensor = torch.from_numpy(mask_np).long()
        return img_tensor, mask_tensor

# ---------------------------------------------------------
# Main 测试函数：验证 8 的倍数对齐问题
# ---------------------------------------------------------
if __name__ == "__main__":
    import shutil
    print("=== [GPRDataset 8倍数对齐功能验证] ===")
    
    test_dir = Path("test_data_alignment")
    imgs_dir = test_dir / "images"
    masks_dir = test_dir / "masks"
    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # 1. 模拟一个非 8 倍数的数据量（比如只有 10 张图）
    num_samples = 10 
    print(f"创建模拟数据量: {num_samples} 张图")
    for i in range(num_samples):
        dummy_img = Image.new('RGB', (64, 64), color=(i, 100, 100))
        dummy_mask = Image.new('L', (64, 64), color=(0))
        dummy_img.save(imgs_dir / f"test_{i:02d}.png")
        dummy_mask.save(masks_dir / f"test_{i:02d}.png")

    try:
        # 2. 初始化 Dataset (默认 align_size=8)
        val_ds = GPRDataset(img_dir=imgs_dir, ann_dir=masks_dir, split="val")
        
        print(f"Dataset 原始数据量 (real_len): {val_ds.real_len}")
        print(f"Dataset 补齐后长度 (__len__): {len(val_ds)}")
        
        # 3. 验证逻辑长度是否符合预期
        assert len(val_ds) % 8 == 0, "长度未对齐为 8 的倍数！"
        assert len(val_ds) == 16, f"预期长度 16，实际得到 {len(val_ds)}"

        # 4. 模拟 DataLoader 遍历
        # 即使 batch_size 是 8，也能刚好跑完 2 个完整的 batch，不会留下余数
        batch_size = 8
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        print(f"DataLoader Batch Size: {batch_size}")
        print(f"预计生成 Batch 数量: {len(loader)}")
        
        for batch_idx, (imgs, masks) in enumerate(loader):
            print(f"Batch {batch_idx} | 形状: {imgs.shape}")
            # 最后一个 batch 的形状验证
            assert imgs.shape[0] == batch_size, f"Batch {batch_idx} 大小非 {batch_size}，对齐失败！"

        # 5. 验证采样的数据是否正确（取模补齐）
        # 第 11 个元素 (idx=10) 应该是第 1 个元素的重复 (actual_idx=0)
        img_0 = val_ds[0][0]
        img_10 = val_ds[10][0]
        assert torch.equal(img_0, img_10), "取模逻辑错误，数据未正确重复补齐"

        print("\n✅ 对齐测试通过！现在所有 Batch 都会是 8 的倍数。")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        shutil.rmtree(test_dir)
        print("清理临时文件。")