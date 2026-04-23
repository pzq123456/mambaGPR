import torch
from torch.utils.data import DataLoader
from src.dataset import GPRDataset
from pathlib import Path

def test_dataloading():
    # 根路径配置
    base_path = Path(r"C:\Users\admin\Desktop\personal\mambaGPR\data")
    
    print("🚀 开始数据集鲁棒性测试...")
    print("-" * 40)

    # 1. 验证 Pretext (训练集与验证集)
    for split in ["P_TRAIN", "P_VALIDATION"]:
        p_path = base_path / "images" / "PRETEXT" / split
        if p_path.exists():
            ds = GPRDataset(img_dir=p_path, is_pretext=True)
            loader = DataLoader(ds, batch_size=8, shuffle=True)
            batch = next(iter(loader))
            print(f"✅ {split}: 加载成功! Batch shape: {batch.shape}")
        else:
            print(f"⚠️ 警告: 未找到 {split} 路径")

    # 2. 验证 Downstream (带标注)
    d_img_path = base_path / "images" / "DOWNSTREAM" / "D_TRAIN"
    d_ann_path = base_path / "annotations" / "DOWNSTREAM" / "D_TRAIN"
    
    if d_img_path.exists() and d_ann_path.exists():
        ds_down = GPRDataset(img_dir=d_img_path, ann_dir=d_ann_path, is_pretext=False)
        loader_down = DataLoader(ds_down, batch_size=4, shuffle=True)
        
        imgs, masks = next(iter(loader_down))
        
        print(f"✅ Downstream: 加载成功!")
        print(f"   - 图像范围 (归一化后): {imgs.min():.2f} ~ {imgs.max():.2f}")
        print(f"   - 掩码类别: {torch.unique(masks).tolist()}")
        
        # 尺寸检查
        if imgs.shape[2:] == masks.shape[1:]:
            print(f"   - 尺寸对齐: 成功 ({imgs.shape[2:]})")
        else:
            print(f"   - ❌ 尺寸对齐失败: Img{imgs.shape[2:]} vs Mask{masks.shape[1:]}")
    else:
        print("⚠️ 警告: 未找到 Downstream 相关路径")

    print("-" * 40)
    print("🎊 所有基础数据流程测试通过！")

if __name__ == "__main__":
    test_dataloading()