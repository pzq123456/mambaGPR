import torch.nn as nn
from transformers import Mamba2Config, Mamba2Model
from src.gpr_cross_scan import GPRCrossScan, GPRCrossMerge

class GPRMamba2(nn.Module):
    def __init__(self, 
                 grid_size=(17, 90), # 340/20=17, 720/8=90
                 in_channels=3, 
                 patch_size=(20, 8),  # 非对称 Patch: (纵向20, 横向8)
                 hidden_size=128, 
                 num_layers=4, 
                 num_classes=2,
                 expand=2):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.hidden_size = hidden_size

        # 1. 非对称 Patch Embedding: 显存优化的核心
        # kernel 和 stride 都设为 patch_size，将图像物理切分为细长条
        self.patch_embed = nn.Conv2d(in_channels, hidden_size, 
                                     kernel_size=patch_size, stride=patch_size)

        # 2. GPR 扫描模块
        self.scan = GPRCrossScan(grid_size=grid_size)
        self.merge = GPRCrossMerge(grid_size=grid_size)

        # 3. Mamba2 Encoder
        head_dim = 64
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

        # 4. 预训练重构头 (Pretext): 像素级还原
        # 使用 Linear 映射回每个 Patch 的像素总数，再进行空间还原
        self.reconstruction_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, in_channels * patch_size[0] * patch_size[1])
        )

        # 5. 下游分割头 (Downstream): 结构化还原
        # 使用反卷积将特征图放大回原始 (340, 720)
        self.segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, hidden_size, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_size, num_classes, kernel_size=1)
        )

    def forward(self, x, mode="pretext"):
        B, C, H, W = x.shape
        
        # 1. 映射与展平
        x_emb = self.patch_embed(x)        # (B, hidden_size, 17, 180)
        curr_h, curr_w = x_emb.shape[2:]
        x_flat = x_emb.flatten(2).transpose(1, 2)  # (B, L, hidden_size)
        
        # 2. 跨扫描与 Mamba 处理
        x_scanned = self.scan(x_flat)
        output = self.encoder(inputs_embeds=x_scanned)
        out = output.last_hidden_state
        x_merged = self.merge(out)         # (B, L, hidden_size)

        if mode == "pretext":
            # --- 重构路径 ---
            recon = self.reconstruction_head(x_merged) # (B, L, C*P_h*P_w)
            # 还原为 (B, curr_h, curr_w, P_h, P_w, C)
            recon = recon.view(B, curr_h, curr_w, self.patch_size[0], self.patch_size[1], C)
            # 维度换位并拼接回完整图像
            recon = recon.permute(0, 5, 1, 3, 2, 4).contiguous()
            recon = recon.view(B, C, H, W)
            return recon

        elif mode == "downstream":
            # --- 分割路径 ---
            feat = x_merged.transpose(1, 2).reshape(B, self.hidden_size, curr_h, curr_w)
            mask = self.segmentation_head(feat)
            return mask