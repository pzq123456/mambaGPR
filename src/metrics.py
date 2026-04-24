# src/metrics.py

import torch
from torchmetrics.classification import (
    JaccardIndex,
    MulticlassPrecision,
    MulticlassRecall,
    BinaryPrecision,
    BinaryRecall
)

from torchmetrics.segmentation import DiceScore

class SegmentationMetrics:

    def __init__(
        self,
        num_classes: int,
        average: str = 'macro',
        ignore_index: int = None,
        threshold: float = 0.5,
        device: str = 'cpu'
    ):
        self.num_classes = num_classes
        self.average = average
        self.ignore_index = ignore_index
        self.threshold = threshold
        self.device = device
        
        # 判断是否为纯二分类（单通道）
        self.is_pure_binary = (num_classes == 1)
        
        self._init_metrics()
        
    def _init_metrics(self):

        task = "binary" if self.is_pure_binary else "multiclass"
        self.jaccard = JaccardIndex(
            task=task,
            num_classes=None if self.is_pure_binary else self.num_classes,
            threshold=self.threshold if self.is_pure_binary else None,
            average=self.average if not self.is_pure_binary else None,
            ignore_index=self.ignore_index
        ).to(self.device)
        
        # 2. mDice (根据 GitHub 文档，DiceScore 在 segmentation 下且无 task 参数)
        # 参数对齐：num_classes, average, input_format='index' (对应我们的 update 逻辑)
        self.dice = DiceScore(
            num_classes=self.num_classes if not self.is_pure_binary else 2, # 二分类在 DiceScore 中通常设为 2 类
            average=self.average,
            input_format='index' # 我们的 update 会把 logits 转为 index
        ).to(self.device)
        
        # 3. Precision & Recall (属于 classification)
        if self.is_pure_binary:
            self.precision = BinaryPrecision(threshold=self.threshold, ignore_index=self.ignore_index).to(self.device)
            self.recall = BinaryRecall(threshold=self.threshold, ignore_index=self.ignore_index).to(self.device)
        else:
            self.precision = MulticlassPrecision(
                num_classes=self.num_classes, 
                average=self.average, 
                ignore_index=self.ignore_index
            ).to(self.device)
            self.recall = MulticlassRecall(
                num_classes=self.num_classes, 
                average=self.average, 
                ignore_index=self.ignore_index
            ).to(self.device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """处理 Logits 并更新指标"""
        preds = preds.to(self.device)
        targets = targets.to(self.device)

        if preds.dim() == 4:
            if preds.size(1) > 1:
                # 多分类 -> Index (B, H, W)
                preds = torch.argmax(preds, dim=1)
            else:
                # 单通道 -> Index (B, H, W)
                preds = (torch.sigmoid(preds).squeeze(1) > self.threshold).long()
        
        preds = preds.long()
        targets = targets.long()

        # 更新
        self.jaccard.update(preds, targets)
        self.dice.update(preds, targets)
        self.precision.update(preds, targets)
        self.recall.update(preds, targets)
    
    def compute(self) -> dict:
        return {
            'mIoU': self.jaccard.compute().item(),
            'mDice': self.dice.compute().item(),
            'Precision': self.precision.compute().item(),
            'Recall': self.recall.compute().item()
        }
    
    def reset(self):
        self.jaccard.reset()
        self.dice.reset()
        self.precision.reset()
        self.recall.reset()

class GPRMetrics(SegmentationMetrics):
    def __init__(self, num_classes=2, ignore_index=None, device='cpu'):
        super().__init__(
            num_classes=num_classes,
            average='macro', 
            ignore_index=ignore_index,
            device=device
        )

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metrics = GPRMetrics(device=device)
    
    # 模拟数据
    dummy_preds = torch.randn(4, 2, 340, 720).to(device)
    dummy_targets = torch.randint(0, 2, (4, 340, 720)).to(device)
    
    metrics.update(dummy_preds, dummy_targets)
    res = metrics.compute()
    
    print("✅ 成功按照 GitHub 仓库规范运行:")
    for k, v in res.items():
        print(f"{k:12}: {v:.4f}")