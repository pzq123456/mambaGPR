# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from pathlib import Path
import traceback
import time

from src.dataset import GPRDataset
from src.model import GPRMamba2
from src.config import config

def run_verify_train(mode="pretext"):
    accelerator = Accelerator(mixed_precision=config.MIXED_PRECISION)
    start_time = time.perf_counter()

    if mode == "pretext":
        img_dir = config.PRETEXT_IMG_DIR
        ann_dir = None
        is_pretext = True
        criterion = nn.MSELoss()
        print(f"🛠️ [Pretext] 启动验证... 路径: {img_dir}")
    else:
        img_dir = config.DOWNSTREAM_IMG_DIR
        ann_dir = config.DOWNSTREAM_ANN_DIR
        is_pretext = False
        criterion = nn.CrossEntropyLoss()
        print(f"🎯 [Downstream] 启动验证... 路径: {img_dir}")

    if not img_dir.exists():
        print(f"⚠️ 找不到目录: {img_dir}")
        return

    # 数据集和 DataLoader
    dataset = GPRDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        is_pretext=is_pretext,
        mean=config.MEAN,       # 注入均值和标准差
        std=config.STD
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 初始化模型（所有参数从 config 读取）
    model = GPRMamba2(
        grid_size=config.GRID_SIZE,
        patch_size=config.PATCH_SIZE,
        in_channels=config.IN_CHANNELS,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        num_classes=config.NUM_CLASSES,
        expand=config.EXPAND,
        head_dim=config.HEAD_DIM
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    step_start = time.perf_counter()
    for i, batch in enumerate(loader):
        optimizer.zero_grad()

        if mode == "pretext":
            inputs = batch
            outputs = model(inputs, mode="pretext")
            loss = criterion(outputs, inputs)
        else:
            inputs, targets = batch
            outputs = model(inputs, mode="downstream")
            loss = criterion(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()

        if i + 1 >= config.MAX_BATCHES:   # 运行指定 batch 数后退出
            break

    accelerator.wait_for_everyone()
    end_time = time.perf_counter()

    total_duration = end_time - start_time
    avg_step_duration = (end_time - step_start) / config.MAX_BATCHES
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    print(f"✅ {mode} 验证成功!")
    print(f"⏱️  总耗时: {total_duration:.2f}s")
    print(f"⏱️  平均 Step 耗时: {avg_step_duration:.4f}s")
    print(f"💾 峰值显存占用: {max_mem:.2f} MB")
    print(f"📉 最终 Loss: {loss.item():.6f}")

if __name__ == "__main__":
    print(f"🚀 环境启动中...")
    print(f"📍 项目根目录: {config.PROJECT_ROOT}")

    try:
        run_verify_train(mode="pretext")
    except Exception:
        print(f"❌ Pretext 验证失败: \n{traceback.format_exc()}")

    print("-" * 60)

    try:
        run_verify_train(mode="downstream")
    except Exception:
        print(f"❌ Downstream 验证失败: \n{traceback.format_exc()}")