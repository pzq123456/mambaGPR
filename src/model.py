# src\model.py

import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2Model
from src.gpr_cross_scan import GPRCrossScan, GPRCrossMerge

class GPRMamba2(nn.Module):
    """
    GPR 专用 Mamba2 模型：支持跨扫描(Cross Scan)机制。
    已修复 Naive Implementation 下的 repeat_interleave 报错。
    """
    def __init__(self, 
                 grid_size=(17, 18), 
                 in_channels=3, 
                 hidden_size=128, 
                 num_layers=4, 
                 num_classes=2,
                 expand=2):
        super().__init__()
        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.hidden_size = hidden_size

        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, kernel_size=1)

        # 2. GPR 扫描与融合模块
        self.scan = GPRCrossScan(grid_size=grid_size)
        self.merge = GPRCrossMerge(grid_size=grid_size)

        # 3. Mamba2 Encoder 
        # 核心约束：hidden_size * expand = num_heads * head_dim
        head_dim = 64
        num_heads = (hidden_size * expand) // head_dim
        
        config = Mamba2Config(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads, 
            head_dim=head_dim,
            expand=expand,
            n_groups=1,       # 显式设置为 1，修复 repeat_interleave 报错
            use_bias=True,
            use_conv_bias=True
        )
        self.encoder = Mamba2Model(config)

        # 4. 预训练重构头
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_channels)
        )

        # 5. 下游分割头
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        )

    def forward(self, x, mode="pretext"):
        B, C, H, W = x.shape
        
        # 1. 映射与展平
        x = self.patch_embed(x)             # (B, hidden_size, H, W)
        x = x.flatten(2).transpose(1, 2)    # (B, L, hidden_size)
        
        # 2. 跨扫描 (四方向并行)
        x_scanned = self.scan(x)            # (4*B, L, hidden_size)
        
        # 3. Mamba2 处理
        # 使用 inputs_embeds 绕过词嵌入层
        output = self.encoder(inputs_embeds=x_scanned)
        out = output.last_hidden_state      # (4*B, L, hidden_size)
        
        # 4. 方向融合
        x_merged = self.merge(out)          # (B, L, hidden_size)

        if mode == "pretext":
            recon = self.reconstruction_head(x_merged)
            # 还原形状为 (B, C, H, W)
            recon = recon.transpose(1, 2).reshape(B, C, self.H, self.W)
            return recon

        elif mode == "downstream":
            feat = x_merged.transpose(1, 2).reshape(B, self.hidden_size, self.H, self.W)
            mask = self.segmentation_head(feat)
            return mask
        else:
            raise ValueError(f"Unknown mode: {mode}")

if __name__ == "__main__":
    # 使用你指定的 grid_size 验证
    H, W = 16, 16
    model = GPRMamba2(grid_size=(H, W), hidden_size=128, num_layers=2)
    
    # 模拟输入 (Batch=2, RGB, 16, 16)
    dummy_input = torch.randn(2, 3, H, W)
    
    print("开始模型测试...")
    # 1. 测试 Pretext 模式
    recon = model(dummy_input, mode="pretext")
    print(f"✅ Pretext 成功! 输出形状: {recon.shape}") 
    
    # 2. 测试 Downstream 模式
    mask = model(dummy_input, mode="downstream")
    print(f"✅ Downstream 成功! 输出形状: {mask.shape}")