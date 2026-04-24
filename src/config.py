# src/config.py
from pathlib import Path
import time

class Config:
    # ---------- 路径配置 ----------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_DATA_PATH = PROJECT_ROOT / "data"

    PRETEXT_IMG_DIR = BASE_DATA_PATH / "images" / "PRETEXT" / "P_TRAIN"
    DOWNSTREAM_IMG_DIR = BASE_DATA_PATH / "images" / "DOWNSTREAM" / "D_TRAIN"
    DOWNSTREAM_ANN_DIR = BASE_DATA_PATH / "annotations" / "DOWNSTREAM" / "D_TRAIN"

    DOWNSTREAM_VAL_IMG_DIR = BASE_DATA_PATH / "images" / "DOWNSTREAM" / "D_VALIDATION"
    DOWNSTREAM_VAL_ANN_DIR = BASE_DATA_PATH / "annotations" / "DOWNSTREAM" / "D_VALIDATION"

    # ---------- 数据参数 ----------
    IMAGE_SIZE = (340, 720)       # 输入图像尺寸
    PATCH_SIZE = (20, 8)          # 非对称 patch: (纵向20, 横向8)
    IN_CHANNELS = 3               # 输入图像通道数
    NUM_CLASSES = 2               # 分割类别数（背景+目标）

    @property
    def GRID_SIZE(self): 
        # 根据 IMAGE_SIZE 和 PATCH_SIZE 自动计算网格尺寸
        return (
            self.IMAGE_SIZE[0] // self.PATCH_SIZE[0],
            self.IMAGE_SIZE[1] // self.PATCH_SIZE[1],
        )

    # ---------- 模型参数 ----------
    HIDDEN_SIZE = 128             # Mamba2 隐藏维度
    NUM_LAYERS = 2                # Mamba2 编码器层数
    EXPAND = 2                    # Mamba2 扩展因子
    HEAD_DIM = 64                 # Mamba2 每头维度

    # ---------- 训练参数 ----------
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    MIXED_PRECISION = "bf16"      # 可设为 "fp16" 或 "no"
    MAX_BATCHES = 2               # 快速验证时最多跑几个 batch (仅用于测试脚本)

    # ---------- 预处理参数 ----------
    MEAN = [0.499, 0.499, 0.499]  # RGB 均值
    STD = [0.085, 0.085, 0.085]   # RGB 标准差

    # ---------- 实验管理 ----------
    WANDB_PROJECT = "GPR-Mamba-Project"
    RESUME = False                  # 是否尝试从断点恢复
    CHECKPOINT_DIR = "checkpoints"   # 权重保存目录
    
    def __init__(self):
        # 动态生成运行名称，确保唯一性
        self.WANDB_RUN_NAME = f"Mamba2-GPR-{time.strftime('%m%d-%H%M')}"

    def to_dict(self):
        """
        将配置转换为可 JSON 序列化的字典。
        1. 使用 dir(self) 获取所有成员（包含 property 的计算结果）。
        2. 过滤掉私有变量、函数和非大写配置项。
        3. 将 Path 对象转换为字符串。
        """
        config_dict = {}
        for key in dir(self):
            # 仅记录大写字母开头的配置项 (约定俗成的常量配置)
            if key.isupper() and not key.startswith("_"):
                value = getattr(self, key)
                # 处理 Path 对象，防止 JSON 序列化失败
                if isinstance(value, Path):
                    config_dict[key] = str(value)
                else:
                    config_dict[key] = value
        return config_dict

# 方便导入使用
config = Config()

if __name__ == "__main__":
    # 快速验证 to_dict 是否正常工作
    print("--- Config 序列化测试 ---")
    d = config.to_dict()
    for k, v in d.items():
        print(f"{k}: {v} (Type: {type(v).__name__})")