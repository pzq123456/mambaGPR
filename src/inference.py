# src/inference.py
import argparse
import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from src.config import config
from src.dataset import GPRDataset
from src.model import GPRMambaIndustrial
from src.metrics import GPRMetrics


def save_simple_comparison(img_path, gt_path, pred_mask, save_path):
    """
    两联图极简可视化：[底图 + 红色真值] | [底图 + 绿色预测值]
    修复了 MCG GPR 数据集标签像素值[0, 1]导致渲染失效的问题，确保高对比度呈现。
    """
    # 1. 载入原始底图并获取原始物理分辨率 (如 720x340)
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    # 2. 生成底图 + 真值叠加 (固定鲜艳红)
    gt_pil_raw = Image.open(gt_path).convert("L").resize((W, H), Image.NEAREST)
    gt_np = np.array(gt_pil_raw)
    
    # 执行工程化容错
    gt_binary_np = (gt_np > (128 if gt_np.max() > 1 else 0)).astype(np.uint8) * 255
    gt_mask = Image.fromarray(gt_binary_np)
    
    gt_color = Image.new("RGB", (W, H), (255, 0, 0)) # 纯红图层
    gt_overlay = Image.composite(Image.blend(img, gt_color, 0.45), img, gt_mask)

    # 3. 生成底图 + 预测值叠加 (固定鲜艳绿)
    pred_mask_pil = Image.fromarray((pred_mask * 255).astype('uint8')).convert("L").resize((W, H), Image.NEAREST)
    pred_color = Image.new("RGB", (W, H), (0, 255, 0)) # 纯绿图层
    pred_overlay = Image.composite(Image.blend(img, pred_color, 0.45), img, pred_mask_pil)

    # 4. 横向无缝拼接两张图
    combined = Image.new("RGB", (W * 2, H))
    combined.paste(gt_overlay, (0, 0))
    combined.paste(pred_overlay, (W, 0))
    combined.save(save_path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="MCG GPR Mini Visual Verification Tool")
    parser.add_argument("--index", type=int, default=0, help="单张图片查询的索引")
    parser.add_argument("--num_samples", type=int, default=None, help="批量抽检前 N 个数据 (如: 5)")
    parser.add_argument("--files", type=str, nargs="+", default=None, help="指定某些特定文件名进行检查 (如: cc210 cc240)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 挂载有监督数据集
    val_dataset = GPRDataset(config.DOWNSTREAM_VAL_IMG_DIR, config.DOWNSTREAM_VAL_ANN_DIR, split="val")
    
    # ---- 核心修改：动态判定和解析索引映射 ----
    if args.files is not None:
        indices = []
        # 将输入统一转为小写，去除可能的扩展名（如 .png），方便不区分大小写地精准配对
        target_stems = [f.lower().split('.')[0] for f in args.files]
        
        # 遍历数据集，通过文件名反查 index
        for idx, img_path in enumerate(val_dataset.img_files):
            if img_path.stem.lower() in target_stems:
                indices.append(idx)
        
        if not indices:
            print(f"❌ 错误: 在验证集中未匹配到输入的任何文件: {args.files}")
            return
        print(f"🎬 模式: 点名抽检模式，匹配成功 {len(indices)} 个文件。")
        
    elif args.num_samples is not None:
        indices = list(range(min(args.num_samples, len(val_dataset))))
        print(f"🎬 模式: 批量检查验证集前 {len(indices)} 个数据")
    else:
        if args.index >= len(val_dataset):
            raise IndexError(f"❌ 索引越界！当前验证集共有 {len(val_dataset)} 张图片。")
        indices = [args.index]
        print(f"🎬 模式: 单张检查样本 [Index: {args.index}]")

    # 2. 模型搭建与权重载入 (只需初始化一次)
    model = GPRMambaIndustrial(config.IN_CHANNELS, config.NUM_CLASSES, config.HIDDEN_SIZE, config.NUM_LAYERS).to(device)
    ckpt_path = config.PROJECT_ROOT / "checkpoints" / "checkpoint_best" / "model.safetensors"
    
    state_dict = load_file(str(ckpt_path))
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict.items()})
    model.eval()

    # 创建输出目录
    out_dir = config.PROJECT_ROOT / "inference_results"
    out_dir.mkdir(exist_ok=True)

    # 3. 循环迭代推理
    for idx in indices:
        img_tensor, mask_tensor = val_dataset[idx]
        img_path = val_dataset.img_files[idx]
        gt_path = val_dataset.ann_dir / img_path.name

        # 前向推理
        logits = model(img_tensor.unsqueeze(0).contiguous().to(device))
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

        # 计算指标
        metrics = GPRMetrics(num_classes=config.NUM_CLASSES, device=device)
        metrics.update(logits, mask_tensor.unsqueeze(0).to(device))
        
        print(f"\n📊 MCG GPR 验证样本 [Index: {idx}] - {img_path.name}")
        for k, v in metrics.compute().items():
            print(f" 🔹 {k:10}: {v:.4f}")
        
        # 保存双联对比图
        save_path = out_dir / f"{img_path.stem}_dual_comparison.png"
        save_simple_comparison(img_path, gt_path, pred_mask, save_path)
        print(f"💾 渲染完毕 -> inference_results/{save_path.name}")
    
    print("\n✅ 所有指定的验证集数据已处理完毕！\n")


if __name__ == "__main__":
    main()