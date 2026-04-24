# src/config.py

from pathlib import Path

class Config:
    # ---------- 路径配置 ----------
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    BASE_DATA_PATH = PROJECT_ROOT / "data"

    PRETEXT_IMG_DIR = BASE_DATA_PATH / "images" / "PRETEXT" / "P_TRAIN"
    DOWNSTREAM_IMG_DIR = BASE_DATA_PATH / "images" / "DOWNSTREAM" / "D_TRAIN"
    DOWNSTREAM_ANN_DIR = BASE_DATA_PATH / "annotations" / "DOWNSTREAM" / "D_TRAIN"

    # ---------- 数据参数 ----------
    IMAGE_SIZE = (340, 720)       # 输入图像尺寸
    PATCH_SIZE = (20, 8)          # 非对称 patch: (纵向20, 横向8)
    IN_CHANNELS = 3               # 输入图像通道数
    NUM_CLASSES = 2               # 分割类别数（背景+目标）

    @property
    def GRID_SIZE(self): # 根据 IMAGE_SIZE 和 PATCH_SIZE 自动计算网格尺寸
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
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-4
    MIXED_PRECISION = "bf16"      # 可设为 "fp16" 或 "no"
    MAX_BATCHES = 2               # 快速验证时最多跑几个 batch

    # ---------- 预处理参数 ----------
    MEAN = [0.499, 0.499, 0.499]  # RGB 均值
    STD = [0.085, 0.085, 0.085]   # RGB 标准差

# 方便导入使用
config = Config()