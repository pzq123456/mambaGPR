# src/engine.py
import torch
import shutil
import os
from pathlib import Path
from accelerate import Accelerator

# ---------------------------------------------------------
# 1. 实验驱动引擎 (Engine)
# ---------------------------------------------------------
class ExperimentEngine:
    """
    负责：WandB 日志、断点保存(Resume)、早停控制(Early Stopping)
    """
    def __init__(self, accelerator: Accelerator, custom_config):
        self.accelerator = accelerator
        self.cfg = custom_config
        self.checkpoint_dir = Path(self.cfg.CHECKPOINT_DIR)
        
        # 仅在主进程创建目录
        if self.accelerator.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 早停计数器
        self.patience = 10
        self.counter = 0
        self.best_score = -float('inf')
        self.early_stop = False

        # 初始化 wandb (由 accelerate 统一管理)
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.cfg.WANDB_PROJECT,
                config=self.cfg.to_dict(),
                init_kwargs={"wandb": {"name": self.cfg.WANDB_RUN_NAME}}
            )

    def log(self, metrics: dict, step: int):
        """记录指标"""
        self.accelerator.log(metrics, step=step)

    def save_and_check_stop(self, current_score: float, step: int):
        """保存状态并检查早停"""
        improved = False
        if current_score > self.best_score:
            self.best_score = current_score
            self.counter = 0
            improved = True
        else:
            self.counter += 1

        # 1. 保存最新的状态 (用于断点续训)
        last_path = self.checkpoint_dir / "checkpoint_last"
        # accelerate 会处理分布式保存，无需手动判断主进程
        self.accelerator.save_state(last_path)

        # 2. 如果性能提升，备份一份作为 Best
        if improved:
            best_path = self.checkpoint_dir / "checkpoint_best"
            if self.accelerator.is_main_process:
                if best_path.exists(): shutil.rmtree(best_path)
                shutil.copytree(last_path, best_path)
            self.accelerator.print(f"✨ 发现更优模型 (Step {step}): {current_score:.4f}，已备份。")

        # 3. 检查早停
        if self.counter >= self.patience:
            self.early_stop = True
            self.accelerator.print(f"🛑 触发早停条件：连续 {self.patience} 次未提升。")

    def load_resume(self):
        """加载最近一次保存的状态"""
        path = self.checkpoint_dir / "checkpoint_last"
        if path.exists():
            self.accelerator.load_state(path)
            self.accelerator.print(f"🔄 成功恢复断点状态: {path}")
            return True
        return False

# ---------------------------------------------------------
# 2. 验证代码 (Main)
# ---------------------------------------------------------
if __name__ == "__main__":
    # 为了让这个脚本能独立运行进行测试，我们临时定义一个 MockConfig
    class MockConfig:
        WANDB_PROJECT = "Test_Project"
        WANDB_RUN_NAME = "Test_Run"
        CHECKPOINT_DIR = "./test_checkpoints"
        def to_dict(self): return {"lr": 1e-4}

    # 初始化配置
    mock_cfg = MockConfig()
    
    # 初始化 Accelerator (注意指定 log_with='wandb')
    # 如果你本地没登录 wandb，可以设置 os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_MODE"] = "offline" 
    accelerator = Accelerator(log_with="wandb")

    # 模拟模型和优化器（必须 prepare 后才能 save_state）
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)

    # 实例化引擎
    engine = ExperimentEngine(accelerator, mock_cfg)

    print("\n--- [测试 1: 日志记录] ---")
    engine.log({"val/mIoU": 0.5}, step=1)

    print("\n--- [测试 2: 保存与提升] ---")
    engine.save_and_check_stop(current_score=0.85, step=10)

    print("\n--- [测试 3: 模拟断点恢复] ---")
    new_engine = ExperimentEngine(accelerator, mock_cfg)
    if new_engine.load_resume():
        print("✅ 断点恢复测试成功")

    print("\n--- [测试 4: 早停触发] ---")
    engine.patience = 2
    engine.save_and_check_stop(0.7, step=20) # 没提升
    engine.save_and_check_stop(0.7, step=30) # 没提升 -> 触发
    print(f"早停状态: {engine.early_stop}")

    # 清理 W&B
    accelerator.end_training()
    # 清理测试文件夹 (可选)
    # shutil.rmtree(mock_cfg.CHECKPOINT_DIR)