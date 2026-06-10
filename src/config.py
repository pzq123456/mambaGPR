# src/config.py
from pathlib import Path
import time

class Config:
    # ---------- 路径配置 ----------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_DATA_PATH = PROJECT_ROOT / "data"

    DOWNSTREAM_IMG_DIR = BASE_DATA_PATH / "images" / "DOWNSTREAM" / "D_TRAIN"
    DOWNSTREAM_ANN_DIR = BASE_DATA_PATH / "annotations" / "DOWNSTREAM" / "D_TRAIN"

    DOWNSTREAM_VAL_IMG_DIR = BASE_DATA_PATH / "images" / "DOWNSTREAM" / "D_VALIDATION"
    DOWNSTREAM_VAL_ANN_DIR = BASE_DATA_PATH / "annotations" / "DOWNSTREAM" / "D_VALIDATION"

    # ---------- 数据参数 ----------
    IMAGE_SIZE = (340, 720)       # 输入雷达图像尺寸
    IN_CHANNELS = 3               # 输入图像通道数
    NUM_CLASSES = 2               # 分割类别数（背景 + 目标物）

    # ---------- 模型参数 ----------
    HIDDEN_SIZE = 128             # Mamba2 隐藏层维度
    NUM_LAYERS = 2                # Mamba2 编码器层数

    # ---------- 训练参数 ----------
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    MIXED_PRECISION = "bf16"      # 可设为 "fp16" 或 "no"
    EPOCHS = 200

    # ---------- 预处理参数 ----------
    MEAN = [0.499, 0.499, 0.499]  # GPR 图像均值
    STD = [0.085, 0.085, 0.085]   # GPR 图像标准差

    # ---------- 实验管理 ----------
    WANDB_PROJECT = "GPR-Mamba-Industrial"
    RESUME = False                  # 是否尝试从断点恢复
    CHECKPOINT_DIR = "checkpoints"   # 权重保存目录
    
    def __init__(self):
        self.WANDB_RUN_NAME = f"U-Mamba2-GPR-{time.strftime('%m%d-%H%M')}"

    def to_dict(self):
        config_dict = {}
        for key in dir(self):
            if key.isupper() and not key.startswith("_"):
                value = getattr(self, key)
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict

config = Config()

if __name__ == "__main__":
    # 快速验证 to_dict 是否正常工作
    print("--- Config 序列化测试 ---")
    d = config.to_dict()
    for k, v in d.items():
        print(f"{k}: {v} (Type: {type(v).__name__})")