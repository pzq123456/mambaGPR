# src/model.py
import torch
import torch.nn as nn
from transformers import Mamba2Config, Mamba2Model
from src.gpr_cross_scan import GPRCrossScan, GPRCrossMerge

class GPRMamba2(nn.Module):
    def __init__(self,
                 grid_size=(17, 90),
                 in_channels=3,
                 patch_size=(20, 8),
                 hidden_size=128,
                 num_layers=4,
                 num_classes=2,
                 expand=2,
                 head_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.hidden_size = hidden_size

        # 1. 非对称 Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, hidden_size - 1,
                                     kernel_size=patch_size, stride=patch_size)

        # 2. GPR 跨扫描模块
        self.scan = GPRCrossScan(grid_size=grid_size)
        self.merge = GPRCrossMerge(grid_size=grid_size)

        # 3. Mamba2 配置（使用传入的 head_dim 和 expand）
        num_heads = (hidden_size * expand) // head_dim
        config = Mamba2Config(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            expand=expand,
            n_groups=1,
            use_bias=True,
            use_conv_bias=True
        )
        self.encoder = Mamba2Model(config)

        # 4. 预训练重构头
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_channels * patch_size[0] * patch_size[1])
        )

        # 5. 下游分割头
        up_h1, up_w1 = patch_size[0] // 2, patch_size[1] // 2
        self.segmentation_head = nn.Sequential(
            nn.Upsample(scale_factor=(up_h1, up_w1), mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size, hidden_size // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(hidden_size // 2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x, mode="pretext"):
        B, C, H, W = x.shape

        # Patch Embedding
        x_emb = self.patch_embed(x)
        curr_h, curr_w = x_emb.shape[2:]

        # 深度增益拼接
        depth_gain = torch.linspace(0, 1, curr_h, device=x.device)
        depth_gain = depth_gain.view(1, 1, curr_h, 1).expand(B, 1, curr_h, curr_w)
        x_emb = torch.cat([x_emb, depth_gain], dim=1)   # (B, hidden_size, curr_h, curr_w)

        # 展平为序列
        x_flat = x_emb.flatten(2).transpose(1, 2)       # (B, L, hidden_size)

        # 跨扫描 + Mamba
        x_scanned = self.scan(x_flat)
        output = self.encoder(inputs_embeds=x_scanned)
        out = output.last_hidden_state
        x_merged = self.merge(out)                      # (B, L, hidden_size)

        if mode == "pretext":
            recon = self.reconstruction_head(x_merged)
            recon = recon.view(B, curr_h, curr_w, self.patch_size[0], self.patch_size[1], C)
            recon = recon.permute(0, 5, 1, 3, 2, 4).contiguous()
            recon = recon.view(B, C, H, W)
            return recon

        elif mode == "downstream":
            feat = x_merged.transpose(1, 2).reshape(B, self.hidden_size, curr_h, curr_w)
            mask = self.segmentation_head(feat)
            return mask