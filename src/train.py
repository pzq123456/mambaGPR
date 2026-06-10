# src/train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm.auto import tqdm
import traceback

from src.dataset import GPRDataset
from src.model import GPRMambaIndustrial
from src.config import config
from src.metrics import GPRMetrics            # 沿用你原本项目的组件
from src.engine import ExperimentEngine        # 沿用你原本项目的组件

def train_industrial():
    """
    工业级有监督分割主训练逻辑
    """
    # 1. 启动加速器
    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION,
        log_with="wandb"
    )
    device = accelerator.device
    engine = ExperimentEngine(accelerator, config)
    
    accelerator.print(f"🚀 启动 U-Mamba 生产级工程化训练...")
    accelerator.print(f"💻 运行设备: {device} | 运算精度混合模式: {config.MIXED_PRECISION}")

    # 2. 构建标准数据加载器
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

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. 初始化全流水线组件
    model = GPRMambaIndustrial(
        in_channels=config.IN_CHANNELS,
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_metrics = GPRMetrics(num_classes=config.NUM_CLASSES, device=device)
    val_metrics = GPRMetrics(num_classes=config.NUM_CLASSES, device=device)

    # 4. 使用 Accelerate 托管
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    if config.RESUME:
        engine.load_resume()

    # 5. 执行主训练阶段
    for epoch in range(config.EPOCHS):
        # --- 训练逻辑 ---
        model.train()
        train_metrics.reset()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:03d} [Train]", disable=not accelerator.is_local_main_process)
        for step, (images, masks) in enumerate(pbar):
            optimizer.zero_grad()
            
            # 双重保险：确保传入批次的 Stride 的绝对规整
            images = images.contiguous()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            accelerator.backward(loss)
            optimizer.step()
            
            epoch_loss += loss.item()
            train_metrics.update(outputs, masks)
            
            if step % 5 == 0:
                cur_m = train_metrics.compute()
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "mIoU": f"{cur_m['mIoU']:.3f}",
                    "Dice": f"{cur_m['mDice']:.3f}"
                })

        # --- 验证评估阶段 ---
        model.eval()
        val_metrics.reset()
        val_loss = 0
        
        # 工业落地建议：在 inference 期间使用标准 no_grad 配合 contiguous 进行双重稳健性卡点
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Validating", disable=not accelerator.is_local_main_process):
                images = images.contiguous()
                
                outputs = model(images)
                loss_v = criterion(outputs, masks)
                
                val_loss += loss_v.item()
                val_metrics.update(outputs, masks)

        # 6. 计算和同步各卡指标
        train_res = train_metrics.compute()
        val_res = val_metrics.compute()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # 7. 日志分发
        engine.log({
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            **{f"train/{k}": v for k, v in train_res.items()},
            **{f"val/{k}": v for k, v in val_res.items()}
        }, step=epoch)

        # 控制台美化输出
        accelerator.print(f"\n📊 [总结] Epoch {epoch:03d}:")
        accelerator.print(f"  [Train] Loss: {avg_train_loss:.4f} | mIoU: {train_res['mIoU']:.4f} | Dice: {train_res['mDice']:.4f}")
        accelerator.print(f"  [Val]   Loss: {avg_val_loss:.4f} | mIoU: {val_res['mIoU']:.4f} | Dice: {val_res['mDice']:.4f}")
        accelerator.print(f"  [Val]   Precision: {val_res['Precision']:.4f} | Recall: {val_res['Recall']:.4f}")

        # 早停与最佳模型权重持久化
        engine.save_and_check_stop(current_score=val_res['mIoU'], step=epoch)
        if engine.early_stop:
            accelerator.print(f"🛑 检测到验证指标不再增长，触发早停机制，训练提前关闭。")
            break

    engine.accelerator.end_training()
    accelerator.print("✅ 工业混合架构模型训练流圆满完成。")

if __name__ == "__main__":
    try:
        train_industrial()
    except Exception:
        print(f"❌ 运行崩溃崩溃，详细 traceback 信息见下: \n{traceback.format_exc()}")