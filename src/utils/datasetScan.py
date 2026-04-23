#  扫描数据集的工具函数，分析图像和标注的基本属性，为后续的数据预处理和模型设计提供参考。

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def analyze_generic_dataset(img_dir, ann_dir=None, name="Dataset"):
    img_files = sorted(os.listdir(img_dir))
    has_ann = ann_dir is not None
    
    if has_ann:
        ann_files = sorted(os.listdir(ann_dir))
        assert len(img_files) == len(ann_files), f"[{name}] 警告：图/标数量不一致！"

    widths, heights = [], []
    img_modes = set()
    ann_modes = set()
    ann_unique_values = set()
    
    # 用于精确计算均值和标准差（基于所有像素，而不仅是图片平均值）
    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    total_pixels = 0

    print(f"\n🚀 正在深度扫描 [{name}] (共 {len(img_files)} 样本)...")
    
    for f in tqdm(img_files):
        img_path = os.path.join(img_dir, f)
        
        with Image.open(img_path) as img:
            widths.append(img.size[0])
            heights.append(img.size[1])
            img_modes.add(img.mode)
            
            # 将图片转为 [0, 1] 的 numpy 数组进行统计
            img_np = np.array(img).astype(float) / 255.0
            
            # 兼容灰度图扩充为 3 通道的情况
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np]*3, axis=-1)
            
            pixel_sum += np.sum(img_np, axis=(0, 1))
            pixel_sq_sum += np.sum(np.square(img_np), axis=(0, 1))
            total_pixels += img_np.shape[0] * img_np.shape[1]

        if has_ann:
            ann_path = os.path.join(ann_dir, f)
            with Image.open(ann_path) as ann:
                ann_modes.add(ann.mode)
                ann_np = np.array(ann)
                ann_unique_values.update(np.unique(ann_np).tolist())

    # 计算全局均值和标准差
    mean = pixel_sum / total_pixels
    std = np.sqrt((pixel_sq_sum / total_pixels) - np.square(mean))

    print("-" * 50)
    print(f"📊 [{name}] 分析报告:")
    print(f"- 尺寸一致性: {'一致' if len(set(zip(widths, heights)))==1 else '不一致'}")
    print(f"- 常用尺寸: {widths[0]}x{heights[0]}")
    print(f"- 图像模式: {img_modes}")
    print(f"- 归一化参数 (Mean): {mean.round(4)}")
    print(f"- 归一化参数 (Std):  {std.round(4)}")
    
    if has_ann:
        print(f"- 标注模式: {ann_modes}")
        print(f"- 类别标签: {sorted(list(ann_unique_values))}")
        # 简单计算背景和前景比例
        # (可选：在循环里统计 0/1 像素个数)
    else:
        print("- 标注情况: 无标注 (Pretext 模式)")
    print("-" * 50)

# --- 配置路径 ---
base_path = r"C:\Users\admin\Desktop\personal\mambaGPR\data"

# 1. 分析下游任务训练集
analyze_generic_dataset(
    os.path.join(base_path, "images", "DOWNSTREAM", "D_TRAIN"),
    os.path.join(base_path, "annotations", "DOWNSTREAM", "D_TRAIN"),
    name="Downstream_Train"
)

# 2. 分析巨大的自监督预训练集 (关键！)
analyze_generic_dataset(
    os.path.join(base_path, "images", "PRETEXT", "P_TRAIN"),
    ann_dir=None,
    name="Pretext_Huge_Data"
)