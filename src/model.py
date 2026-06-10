# src/model.py
import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2Model

class ConvBlock(nn.Module):
    """工业级双层卷积块 (Conv + BN + ReLU)"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class GPRMambaIndustrial(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size

        # 1. CNN Encoder (提取浅层多尺度特征)
        self.inc = ConvBlock(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, hidden_size))

        # 2. Mamba2 Bottleneck
        # =============== 【核心修复点】 ===============
        # 显式匹配参数，必须严格满足 hidden_size * expand == num_heads * head_dim
        # 128 * 2 == 16 * 16 == 256，且 head_dim=16 完美满足底层硬件的 8 字节对齐
        config = Mamba2Config(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            expand=2,
            head_dim=16,
            num_heads=16,
            n_groups=1,
            use_bias=True,
            use_conv_bias=True
        )
        self.mamba_bottleneck = Mamba2Model(config)
        # =============================================

        # 3. CNN Decoder (恢复分辨率 & 跳跃连接融合特征)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.m_block1 = ConvBlock(hidden_size + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.m_block2 = ConvBlock(128 + 64, 64)

        # 4. 分割预测头
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Encoder 阶段 ---
        x1 = self.inc(x)           # 高频精细局部特征 (B, 64, 340, 720)
        x2 = self.down1(x1)        # 中期降采样特征   (B, 128, 170, 360)
        x3 = self.down2(x2)        # 低维瓶颈特征     (B, 128, 85, 180)

        # --- Mamba2 全局深度长距离上下文捕获 ---
        curr_h, curr_w = x3.shape[2:]
        # 展平为序列，并使用 .contiguous() 确保内存无碎片
        x_flat = x3.flatten(2).transpose(1, 2).contiguous() 
        
        # 核心长序列序列化建模
        mamba_out = self.mamba_bottleneck(inputs_embeds=x_flat).last_hidden_state
        
        # 还原回特征图格式并确保 Stride 干净
        x3_mamba = mamba_out.transpose(1, 2).reshape(B, self.hidden_size, curr_h, curr_w).contiguous()

        # --- Decoder 阶段 (融合浅层高分辨率特征，消除边缘模糊) ---
        x_up = self.up1(x3_mamba)
        if x_up.shape[2:] != x2.shape[2:]:
            x_up = nn.functional.interpolate(x_up, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x_merge = torch.cat([x_up, x2], dim=1)
        x_dec1 = self.m_block1(x_merge)

        x_up2 = self.up2(x_dec1)
        if x_up2.shape[2:] != x1.shape[2:]:
            x_up2 = nn.functional.interpolate(x_up2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        x_merge2 = torch.cat([x_up2, x1], dim=1)
        x_dec2 = self.m_block2(x_merge2)

        # 输出分割图的概率分布 (Logits)
        logits = self.outc(x_dec2)
        return logits

if __name__ == "__main__":
    # 模型前向推理自我测试
    print("--- 工业模型结构测试 ---")
    model = GPRMambaIndustrial(in_channels=3, num_classes=2, hidden_size=128)
    dummy_x = torch.randn(2, 3, 340, 720)
    out = model(dummy_x)
    print(f"输入形状: {dummy_x.shape} -> 输出最终 Mask 形状: {out.shape}")
    assert out.shape == (2, 2, 340, 720), "⚠️ 尺寸不一致，检查上采样层！"
    print("✅ 模型定义成功，前向推理完全正常。")