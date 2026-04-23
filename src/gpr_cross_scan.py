# src\gpr_cross_scan.py
import torch
import torch.nn as nn

class GPRCrossScan(nn.Module):
    def __init__(self, grid_size=(17, 18)):
        super().__init__()
        self.H, self.W = grid_size

    def forward(self, x):
        B, L, C = x.shape
        # 1. 还原 2D 结构: (B, H, W, C)
        x = x.view(B, self.H, self.W, C)
        
        # 方向 1: 水平正向
        x1 = x.reshape(B, -1, C)
        # 方向 2: 水平反向
        x2 = x1.flip(dims=[1])
        # 方向 3: 垂直正向 (关键修正：transpose 后 reshape)
        x3 = x.transpose(1, 2).reshape(B, -1, C)
        # 方向 4: 垂直反向
        x4 = x3.flip(dims=[1])
        
        return torch.cat([x1, x2, x3, x4], dim=0)

class GPRCrossMerge(nn.Module):
    def __init__(self, grid_size=(17, 18)):
        super().__init__()
        self.H, self.W = grid_size

    def forward(self, x):
        B = x.shape[0] // 4
        C = x.shape[-1]
        
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=0)
        
        # 还原方向 2, 4
        x2 = x2.flip(dims=[1])
        x4 = x4.flip(dims=[1])
        
        # 还原方向 1 & 2 的空间结构
        y12 = (x1 + x2) / 2
        y12 = y12.view(B, self.H, self.W, C)
        
        # 还原方向 3 & 4 的空间结构 (先还原到 W, H，再转回 H, W)
        y34 = (x3 + x4) / 2
        y34 = y34.view(B, self.W, self.H, C).transpose(1, 2)
        
        # 最终融合并展平回序列
        return ((y12 + y34) / 2).reshape(B, -1, C)

# --- 严格验证测试 ---
if __name__ == "__main__":
    H, W, C = 17, 18, 192
    scan = GPRCrossScan(grid_size=(H, W))
    merge = GPRCrossMerge(grid_size=(H, W))
    
    dummy_seq = torch.randn(1, H*W, C)
    
    scanned = scan(dummy_seq)
    merged = merge(scanned)
    
    diff = torch.abs(dummy_seq - merged).max()
    print(f"修正后的最大重构误差: {diff.item():.2e}")
    if diff < 1e-5:
        print("✅ 逻辑完美对齐！")
    else:
        print("❌ 逻辑仍有偏差，请检查。")