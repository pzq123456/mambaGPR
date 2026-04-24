# src/train.py
# 用于验证训练流程的正确性和性能指标的计算，确保在正式训练前一切准备就绪。
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
import traceback
import time

from src.dataset import GPRDataset
from src.model import GPRMamba2
from src.config import config
from src.metrics import GPRMetrics

def run_verify_train(mode="pretext"):
    accelerator = Accelerator(mixed_precision=config.MIXED_PRECISION)
    device = accelerator.device  # 获取当前设备
    start_time = time.perf_counter()

    # 初始化指标 (仅在 downstream 模式下使用)
    metrics = None
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
        # 初始化分割指标
        metrics = GPRMetrics(
            num_classes=config.NUM_CLASSES, 
            ignore_index=config.IGNORE_INDEX if hasattr(config, 'IGNORE_INDEX') else None,
            device=device
        )
        print(f"🎯 [Downstream] 启动验证... 路径: {img_dir}")

    if not img_dir.exists():
        print(f"⚠️ 找不到目录: {img_dir}")
        return

    # 数据集和 DataLoader
    dataset = GPRDataset(
        img_dir=img_dir,
        ann_dir=ann_dir,
        is_pretext=is_pretext,
        mean=config.MEAN,
        std=config.STD
    )
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # 初始化模型
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

    # 使用 Accelerate 准备组件
    # 注意：指标对象不需要 accelerator.prepare，因为它内部处理了 device
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
            
            # --- 指标更新 ---
            if metrics is not None:
                # 这里的 outputs 是 logits, targets 是 label index
                metrics.update(outputs, targets)

        accelerator.backward(loss)
        optimizer.step()

        if i + 1 >= config.MAX_BATCHES:
            break

    accelerator.wait_for_everyone()
    end_time = time.perf_counter()

    # 计算最终性能指标
    metric_results = {}
    if metrics is not None:
        metric_results = metrics.compute()
        metrics.reset() # 验证结束后重置

    total_duration = end_time - start_time
    avg_step_duration = (end_time - step_start) / min(len(loader), config.MAX_BATCHES)
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

    print(f"\n✅ {mode} 验证成功!")
    print("-" * 30)
    print(f"⏱️  总耗时: {total_duration:.2f}s")
    print(f"⏱️  平均 Step 耗时: {avg_step_duration:.4f}s")
    print(f"💾 峰值显存占用: {max_mem:.2f} MB")
    print(f"📉 最终 Loss: {loss.item():.6f}")
    
    if metric_results:
        print(f"📊 分割评估指标:")
        for k, v in metric_results.items():
            print(f"   >> {k:10}: {v:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    print(f"🚀 环境启动中...")
    print(f"📍 项目根目录: {config.PROJECT_ROOT}")

    try:
        run_verify_train(mode="pretext")
    except Exception:
        print(f"❌ Pretext 验证失败: \n{traceback.format_exc()}")

    print("\n" + "=" * 60 + "\n")

    try:
        run_verify_train(mode="downstream")
    except Exception:
        print(f"❌ Downstream 验证失败: \n{traceback.format_exc()}")