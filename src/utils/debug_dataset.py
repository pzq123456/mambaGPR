# src/debug_dataset.py
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

# 仅复用路径配置
from src.config import config


def inspect_and_render_gpr_data(img_dir, ann_dir, limit=5):
    """
    完全绕过 PyTorch，直接读取、统计并渲染 MCG GPR 原始数据集。
    将“一片黑”的标签转换为肉眼可见的叠加图。
    """
    img_dir = Path(img_dir)
    ann_dir = Path(ann_dir)
    
    # 1. 扫描文件
    img_files = sorted(list(img_dir.glob("*.png")))
    if not img_files:
        print(f"❌ 错误：在路径 {img_dir} 下未找到任何 .png 雷达图！")
        return

    print(f"\n🚀 开始检测数据集。共找到 {len(img_files)} 张图片。将抽检前 {min(limit, len(img_files))} 张...\n")

    # 创建调试结果输出目录
    output_dir = config.PROJECT_ROOT / "debug_data_outputs"
    output_dir.mkdir(exist_ok=True)

    for idx in range(min(limit, len(img_files))):
        img_path = img_files[idx]
        ann_path = ann_dir / img_path.name

        if not ann_path.exists():
            print(f"⚠️ 警告：图片 {img_path.name} 找不到对应的 annotations 掩码文件！跳过。")
            continue

        # 2. 纯 PIL / NumPy 读取
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")
        
        W, H = img.size
        mask_np = np.array(mask)
        
        # 3. 核心统计参数计算 (Debug 关键)
        max_val = mask_np.max()
        min_val = mask_np.min()
        unique_vals = np.unique(mask_np)
        
        # 动态解析标签：不管是[0, 1]还是[0, 255]，都计算真实的目标物像素点
        target_condition = (mask_np > 128) if max_val > 1 else (mask_np == 1)
        total_pixels = W * H
        target_pixels = np.sum(target_condition)
        target_ratio = (target_pixels / total_pixels) * 100

        # 打印单张图的统计报告
        print(f"================ 📄 样本 [{idx}] 统计数据: {img_path.name} ================")
        print(f" 🔹 原始分辨率 (Width x Height) : {W} x {H}")
        print(f" 🔹 掩码像素值范围 (Min ~ Max)  : {min_val} ~ {max_val}")
        print(f" 🔹 掩码包含的所有唯一像素值   : {unique_vals}")
        print(f" 🔹 目标物（病害区域）像素数   : {target_pixels} 个像素")
        print(f" 🔹 目标物在整张图中面积占比   : {target_ratio:.2f}%")
        if target_pixels == 0:
            print(" 🛑 【注意】这张图的 GT 标注是纯黑的空标签（没有目标物）！")
        print("-----------------------------------------------------------------------")

        # 4. 强制尺寸对齐防错
        if img.size != mask.size:
            mask = mask.resize(img.size, Image.NEAREST)

        # 5. 【核心修复】把看不见的 [0, 1] 转化为强对比可见的 [0, 255] 掩码
        gt_binary_np = target_condition.astype(np.uint8) * 255
        visible_mask = Image.fromarray(gt_binary_np)

        # 6. 结合底图渲染 (真值用半透明红色叠加)
        red_layer = Image.new("RGB", (W, H), (255, 0, 0))
        # Image.composite 使用 255強度的 visible_mask 作为权重，把红层叠到底图上
        overlay_img = Image.composite(Image.blend(img, red_layer, 0.45), img, visible_mask)

        # 7. 左右横向拼接：[原始灰色雷达图] | [叠加了红色标注的渲染图]
        combined = Image.new("RGB", (W * 2, H))
        combined.paste(img, (0, 0))
        combined.paste(overlay_img, (W, 0))

        # 8. 保存检查结果
        save_path = output_dir / f"debug_{img_path.stem}.png"
        combined.save(save_path)
        print(f"💾 渲染图已存至: debug_data_outputs/{save_path.name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPR Dataset Visual Inspection Tool")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="检查验证集(val)还是训练集(train)")
    parser.add_argument("--limit", type=int, default=5, help="限制抽检和渲染的图片数量")
    args = parser.parse_args()

    # 根据参数自动切换路径
    if args.split == "train":
        img_dir = config.DOWNSTREAM_IMG_DIR
        ann_dir = config.DOWNSTREAM_ANN_DIR
    else:
        img_dir = config.DOWNSTREAM_VAL_IMG_DIR
        ann_dir = config.DOWNSTREAM_VAL_ANN_DIR

    inspect_and_render_gpr_data(img_dir, ann_dir, limit=args.limit)