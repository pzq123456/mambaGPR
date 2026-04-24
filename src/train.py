# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import traceback

from src.dataset import GPRDataset
from src.model import GPRMamba2
from src.config import config
from src.metrics import GPRMetrics
from src.engine import ExperimentEngine

def train_downstream():
    """
    正式的有监督训练函数：200 Epoch，带验证集、Log、指标打印
    """
    # 1. 初始化 Accelerator 与 实验引擎
    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION,
        log_with="wandb"
    )
    device = accelerator.device
    engine = ExperimentEngine(accelerator, config)
    
    accelerator.print(f"🚀 启动有监督训练 | 设备: {device} | 精度: {config.MIXED_PRECISION}")

    # 2. 准备数据集 (利用优化后的 split 参数)
    train_dataset = GPRDataset(
        img_dir=config.DOWNSTREAM_IMG_DIR,
        ann_dir=config.DOWNSTREAM_ANN_DIR,
        split="train",
        mean=config.MEAN,
        std=config.STD
    )
    val_dataset = GPRDataset(
        img_dir=config.DOWNSTREAM_VAL_IMG_DIR,
        ann_dir=config.DOWNSTREAM_VAL_ANN_DIR,
        split="val",
        mean=config.MEAN,
        std=config.STD
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. 初始化模型、优化器与损失函数
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
    criterion = nn.CrossEntropyLoss()
    
    # 4. 指标计算器 (训练和验证分开)
    train_metrics = GPRMetrics(num_classes=config.NUM_CLASSES, device=device)
    val_metrics = GPRMetrics(num_classes=config.NUM_CLASSES, device=device)

    # 5. Accelerate 准备
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # 6. 断点恢复检查
    if config.RESUME:
        engine.load_resume()

    # ---------- 核心循环 ----------
    epochs = 200
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_metrics.reset()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", disable=not accelerator.is_local_main_process)
        for step, (images, masks) in enumerate(pbar):
            optimizer.zero_grad()
            
            outputs = model(images, mode="downstream")
            loss = criterion(outputs, masks)
            
            accelerator.backward(loss)
            optimizer.step()
            
            # 记录 Loss 与更新指标
            epoch_loss += loss.item()
            train_metrics.update(outputs, masks)
            
            # 每 10 步在进度条打印一次 4 指标的实时状态
            if step % 10 == 0:
                cur_m = train_metrics.compute()
                pbar.set_postfix({
                    "L": f"{loss.item():.3f}",
                    "mIoU": f"{cur_m['mIoU']:.3f}",
                    "Dice": f"{cur_m['mDice']:.3f}"
                })

        # --- 验证阶段 ---
        model.eval()
        val_metrics.reset()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validating", disable=not accelerator.is_local_main_process):
                outputs = model(images, mode="downstream")
                loss_v = criterion(outputs, masks)
                val_loss += loss_v.item()
                val_metrics.update(outputs, masks)

        # --- 计算最终指标 ---
        train_res = train_metrics.compute()
        val_res = val_metrics.compute()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # --- 打印与日志记录 ---
        # 记录到 WandB
        engine.log({
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            **{f"train/{k}": v for k, v in train_res.items()},
            **{f"val/{k}": v for k, v in val_res.items()}
        }, step=epoch)

        # 控制台打印完整的 4 个指标
        accelerator.print(f"\n📊 Epoch {epoch} 总结:")
        accelerator.print(f"  [Train] Loss: {avg_train_loss:.4f} | mIoU: {train_res['mIoU']:.4f} | Dice: {train_res['mDice']:.4f}")
        accelerator.print(f"  [Val]   Loss: {avg_val_loss:.4f} | mIoU: {val_res['mIoU']:.4f} | Dice: {val_res['mDice']:.4f}")
        accelerator.print(f"  [Val]   Precision: {val_res['Precision']:.4f} | Recall: {val_res['Recall']:.4f}")

        # --- 保存最优模型与早停 ---
        # 我们使用验证集的 mIoU 作为保存 Best 的依据
        engine.save_and_check_stop(current_score=val_res['mIoU'], step=epoch)
        
        if engine.early_stop:
            accelerator.print(f"🛑 触发早停，训练提前结束。")
            break

    engine.accelerator.end_training()
    accelerator.print("✅ 训练任务已完成。")

if __name__ == "__main__":
    # 执行正式的有监督训练
    try:
        train_downstream()
    except Exception:
        print(f"❌ 训练过程中发生错误: \n{traceback.format_exc()}")