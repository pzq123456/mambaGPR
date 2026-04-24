# src\train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from pathlib import Path

# 确保导入你重构后的类
from src.dataset import GPRDataset
from src.model import GPRMamba2

def run_verify_train(mode="pretext"):
    """
    统一验证函数
    mode: "pretext" (自监督重构) 或 "downstream" (监督分割)
    """
    accelerator = Accelerator(mixed_precision="fp16")
    
    # --- 核心路径配置 ---
    base_path = Path(r"C:\Users\admin\Desktop\personal\mambaGPR\data")
    
    # --- 动态配置：非对称 Patch 设计 ---
    patch_h, patch_w = 20, 4  # 与模型中定义的 patch_size 保持一致
    # 原始 340 / 20 = 17, 原始 720 / 4 = 180
    model_grid = (17, 180) 
    batch_size = 2

    if mode == "pretext":
        img_dir = base_path / "images" / "PRETEXT" / "P_TRAIN"
        ann_dir = None
        is_pretext = True
        criterion = nn.MSELoss()
        print(f"🛠️ 正在验证 [Pretext] (非对称Patch): 路径 {img_dir}")
    else:
        img_dir = base_path / "images" / "DOWNSTREAM" / "D_TRAIN"
        ann_dir = base_path / "annotations" / "DOWNSTREAM" / "D_TRAIN"
        is_pretext = False
        criterion = nn.CrossEntropyLoss()
        print(f"🎯 正在验证 [Downstream] (非对称Patch): 路径 {img_dir}")

    # 1. 实例化 Dataset
    dataset = GPRDataset(
        img_dir=img_dir, 
        ann_dir=ann_dir,
        is_pretext=is_pretext
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型 (关键：传入计算好的 model_grid)
    model = GPRMamba2(
        grid_size=model_grid, 
        patch_size=(patch_h, patch_w),
        hidden_size=128, 
        num_layers=2,
        num_classes=2
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Accelerator 包装
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # 4. 跑几个 Batch 验证流程
    model.train()
    total_loss = 0
    
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        
        if mode == "pretext":
            inputs = batch
            # 现在的 model(mode="pretext") 内部已经处理好了从 (B, L, C) 到 (B, 3, 340, 720) 的重构
            outputs = model(inputs, mode="pretext")
            loss = criterion(outputs, inputs) 
        else:
            inputs, targets = batch
            outputs = model(inputs, mode="downstream")
            loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        
        if i >= 1: # 跑 2 个 batch 验证
            break

    print(f"✅ {mode} 验证成功! 最终 Step Loss: {loss.item():.4f}") # type: ignore

if __name__ == "__main__":
    # 执行验证
    print("🚀 开始显存优化版模型验证...")
    try:
        run_verify_train(mode="pretext")
    except Exception as e:
        import traceback
        print(f"❌ Pretext 验证失败: \n{traceback.format_exc()}")

    print("-" * 50)

    try:
        run_verify_train(mode="downstream")
    except Exception as e:
        import traceback
        print(f"❌ Downstream 验证失败: \n{traceback.format_exc()}")