# src\train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from pathlib import Path

# 确保导入你现有的类
from src.dataset import GPRDataset
from src.model import GPRMamba2

def run_verify_train(mode="pretext"):
    """
    统一验证函数
    mode: "pretext" (自监督重构) 或 "downstream" (监督分割)
    """
    accelerator = Accelerator(mixed_precision="fp16")
    
    # --- 核心路径配置 (基于你测试成功的路径) ---
    base_path = Path(r"C:\Users\admin\Desktop\personal\mambaGPR\data")
    
    # --- 动态配置 ---
    # 根据你的测试日志，图像尺寸为 (340, 720)
    img_size = (340, 720) 
    batch_size = 4 

    if mode == "pretext":
        img_dir = base_path / "images" / "PRETEXT" / "P_TRAIN"
        ann_dir = None
        is_pretext = True
        criterion = nn.MSELoss()
        print(f"🛠️ 正在验证 [Pretext]: 路径 {img_dir}")
    else:
        img_dir = base_path / "images" / "DOWNSTREAM" / "D_TRAIN"
        ann_dir = base_path / "annotations" / "DOWNSTREAM" / "D_TRAIN"
        is_pretext = False
        criterion = nn.CrossEntropyLoss()
        print(f"🎯 正在验证 [Downstream]: 路径 {img_dir}")

    # 1. 实例化 Dataset
    dataset = GPRDataset(
        img_dir=img_dir, 
        ann_dir=ann_dir,
        is_pretext=is_pretext
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. 初始化模型 (grid_size 必须匹配 340x720)
    model = GPRMamba2(
        grid_size=img_size, 
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
    
    # 模拟 1 个 epoch 中的前几个 step
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        
        if mode == "pretext":
            # batch 是 tensor: (B, 3, H, W)
            inputs = batch
            outputs = model(inputs, mode="pretext")
            loss = criterion(outputs, inputs) # 重构原图
        else:
            # batch 是 (img_tensor, mask_tensor)
            inputs, targets = batch
            outputs = model(inputs, mode="downstream")
            loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        
        if i >= 1: # 跑 2 个 batch 足够证明流程无误
            break

    print(f"✅ {mode} 验证成功! 最终 Step Loss: {loss.item():.4f}")

if __name__ == "__main__":
    # 执行自监督验证
    try:
        run_verify_train(mode="pretext")
    except Exception as e:
        print(f"❌ Pretext 验证过程中报错: \n{e}")

    print("-" * 50)

    # 执行分割验证
    try:
        run_verify_train(mode="downstream")
    except Exception as e:
        print(f"❌ Downstream 验证过程中报错: \n{e}")