# src/gpr_cross_scan.py
import torch
import torch.nn as nn

class GPRCrossScan(nn.Module):
    def __init__(self, grid_size=(17, 90)):   # 默认值与 model 对齐
        super().__init__()
        self.H, self.W = grid_size

    def forward(self, x):
        B, L, C = x.shape
        x = x.view(B, self.H, self.W, C)

        x1 = x.reshape(B, -1, C)                     # 水平正向
        x2 = x1.flip(dims=[1])                       # 水平反向
        x3 = x.transpose(1, 2).reshape(B, -1, C)     # 垂直正向
        x4 = x3.flip(dims=[1])                       # 垂直反向

        return torch.cat([x1, x2, x3, x4], dim=0)


class GPRCrossMerge(nn.Module):
    def __init__(self, grid_size=(17, 90)):
        super().__init__()
        self.H, self.W = grid_size

    def forward(self, x):
        B = x.shape[0] // 4
        C = x.shape[-1]

        x1, x2, x3, x4 = torch.chunk(x, 4, dim=0)

        x2 = x2.flip(dims=[1])
        x4 = x4.flip(dims=[1])

        y12 = (x1 + x2) / 2
        y12 = y12.view(B, self.H, self.W, C)

        y34 = (x3 + x4) / 2
        y34 = y34.view(B, self.W, self.H, C).transpose(1, 2)

        return ((y12 + y34) / 2).reshape(B, -1, C)


# 自检代码（采用修正后的尺寸）
if __name__ == "__main__":
    H, W, C = 17, 90, 192
    scan = GPRCrossScan(grid_size=(H, W))
    merge = GPRCrossMerge(grid_size=(H, W))

    dummy_seq = torch.randn(1, H * W, C)
    scanned = scan(dummy_seq)
    merged = merge(scanned)

    diff = torch.abs(dummy_seq - merged).max()
    print(f"重构误差: {diff.item():.2e}")
    if diff < 1e-5:
        print("✅ 逻辑完美对齐！")
    else:
        print("❌ 仍有偏差，请检查。")