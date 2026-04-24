# 在 MCG GPR 数据集上 复现 Mamba-Net\


gpr max

## data set
扫描结果非常清晰！作为一个 CV 专家，你一眼就能看出这个数据集的结构其实非常有诚意，它不仅仅给了你用来刷榜的“考卷”，还给了你大量的“课外读物”。

基于你的扫描报告，我为你深度解读一下这个数据集的**隐藏价值**和**实战策略**：

### 1. 数据集核心架构拆解

这个数据集被分成了两个截然不同的阶段，这决定了你如何复现那个 $0.611$ 的结果：

* **DOWNSTREAM（下游任务 - 你的主战场）**
    * **TRAIN (630)** / **VALID (168)** / **TEST (168)**
    * 这就是典型的“带标注”分割数据集。总共约 1000 张图。
    * **复现逻辑：** 你在论文表中看到的 $mIoU=0.611$ 应该就是基于这 630 张图训练，在 168 张 TEST 上跑出来的。
    * **注意：** 数据量确实很小（只有 600 多张训练图），这验证了我之前的猜想：**直接 Scratch 训练必挂，必须用预训练权重。**

* **PRETEXT（前置任务 - 你的“秘密武器”）**
    * **TRAIN (4400)** / **VALID (2734)**
    * **注意：** 这里只有 `images` 目录下有文件，`annotations` 下没有对应的。这说明这 **7000+ 张图是无标注数据**。
    * **专家建议：** 如果你直接跑 VM-UNet，这部分数据暂时用不到。但如果你想做出超越 $0.611$ 的 SOTA，你可以用这 7000 张图先跑一个 **MAE（Masked Autoencoder）自监督训练**。让 Mamba 先学会地下地层的基本构造，再到 630 张图上微调，精度会起飞。

---

### 2. 针对 VM-UNet 复现的目录适配方案

由于 `JCruan519/VM-UNet` 或大多数医学影像仓库通常要求简单的 `train/val/test` 结构，你需要做一个 **Symbolic Link (符号链接)** 或者写个简单的脚本把它们映射过去：

**建议的目录映射逻辑：**
```text
# 你的目标训练目录
GPR_Standard/
├── train/
│   ├── images/  <-- 对应 DOWNSTREAM/D_TRAIN/
│   └── masks/   <-- 对应 DOWNSTREAM/D_TRAIN/ (annotations 下的)
├── val/
│   ├── images/  <-- 对应 DOWNSTREAM/D_VALIDATION/
│   └── masks/   <-- 对应 DOWNSTREAM/D_VALIDATION/
└── test/
    ├── images/  <-- 对应 DOWNSTREAM/D_TEST/
    └── masks/   <-- 对应 DOWNSTREAM/D_TEST/
```

## 配环境步骤

激活 uv 环境
```bash
source .venv/bin/activate
```

### 1. 首先在 宿主机 上 安装 cuda 13 驱动（最新的）然后进入 wsl 中确定可以通过 `nvidia-smi` 看到你的显卡和驱动版本

### 2. 在 wsl 中安装 cuda 12.8 的 toolkit（不安装驱动）

```bash
## 安装 CUDA 12.8 驱动
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run
```

```bash
# 强制将 12.8 放在 PATH 的最前面
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

# 再次验证，必须看到 12.8
nvcc -V
```

### 4. 开始安装
这一步需要指定 gcc-12 的路径。

```bash
CC=gcc-12 CXX=g++-12 \
TORCH_CUDA_ARCH_LIST="10.0" \
MAMBA_FORCE_BUILD=TRUE \
uv pip install \
    --no-binary causal-conv1d \
    --no-binary mamba-ssm \
    --no-cache \
    causal-conv1d mamba-ssm
```

```bash
# 设置环境变量指向 GCC 12，然后开始安装
CC=gcc-12 CXX=g++-12 uv pip install causal-conv1d --no-cache
CC=gcc-12 CXX=g++-12 uv pip install mamba-ssm --no-cache

# 记得带上 GCC 12 的环境变量，除非你确认 12.8 已经原生支持了你的默认 GCC
CC=gcc-12 CXX=g++-12 uv pip install causal-conv1d mamba-ssm --no-cache
```

验证代码：
```bash
uv run accelerate launch -m src.train
```



## 训练流程：
非常赞同你的工程直觉。既然你决定在 `Accelerate` 的基础上构建，这套 **"Accelerate + TorchMetrics + WandB"** 的组合确实是目前兼顾“控制力”与“省心”的最优解。

为了让你更清晰地理解这三者是如何协同工作的，我把整个训练流程拆解为**初始化、训练循环、指标监控、检查点管理**四个维度：

### 1. 整体架构分工
* **Accelerate:** 负责“脏活累活”。它接管设备绑定（GPU/CPU）、混合精度（BF16）、数据分发和梯度反传。
* **TorchMetrics:** 负责“尺子”。它在每个 Step 收集 `preds` 和 `targets`，在验证结束时算出符合论文标准的指标。
* **WandB:** 负责“仪表盘”。它把 Loss 和 Metrics 实时绘图，并把每一轮生成的雷达掩膜（Mask）存成图片供你对比。

---

### 2. 训练全流程详细拆解

#### 第一步：环境与监控初始化
在脚本开头，你需要配置 `Accelerator` 和 `WandB`。
```python
from accelerate import Accelerator
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassDice
import wandb

# 1. 初始化加速器
accelerator = Accelerator(mixed_precision="bf16", log_with="wandb")

# 2. 只有在主进程初始化 WandB (配置实验名称、超参数)
if accelerator.is_main_process:
    accelerator.init_trackers(
        project_name="MambaGPR_MCG",
        config={
            "learning_rate": 1e-4,
            "hidden_size": 128,
            "patch_size": (20, 8),
            "batch_size": 16
        }
    )

# 3. 初始化指标 (放在设备上)
miou_fn = MulticlassJaccardIndex(num_classes=2, average='macro').to(accelerator.device)
```

#### 第二步：训练循环 (The Loop)
相比原生 PyTorch，你只需把模型和数据传给 `accelerator.prepare`，它会自动处理剩下的事。
```python
model, optimizer, train_loader, val_loader = accelerator.prepare(
    model, optimizer, train_loader, val_loader
)

for epoch in range(100):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        outputs = model(inputs, mode="downstream")
        loss = criterion(outputs, targets)
        
        # 使用加速器进行反向传播
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录 Loss 到 WandB
        accelerator.log({"train_loss": loss.item()}, step=global_step)
```

#### 第三步：验证与指标对齐 (Evaluation)
每轮训练结束后，进行定量评估。这时 `TorchMetrics` 登场：
```python
    model.eval()
    miou_fn.reset() # 每一轮开始前重置
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs, mode="downstream")
            preds = torch.argmax(outputs, dim=1)
            
            # 把数据喂给指标函数 (跨显卡自动同步)
            miou_fn.update(preds, targets)
            
        # 计算整轮的最终指标
        final_miou = miou_fn.compute()
        
        # 记录到 WandB
        accelerator.log({"val_miou": final_miou}, step=epoch)
```

#### 第四步：检查点与参数管理 (Checkpointing)
关于**模型参数管理**，`Accelerate` 提供了一个非常稳健的方案：

1.  **保存：** 你不需要手动提取 `state_dict`。使用 `accelerator.save_state("path/to/checkpoint")`。它会保存模型权重、优化器状态、甚至当前的随机数种子。
2.  **加载：** 如果训练中断，使用 `accelerator.load_state("path/to/checkpoint")` 即可无缝恢复。
3.  **最优模型筛选：** 你可以设置一个逻辑，如果当前 `final_miou` 超过了之前的最高分，就调用一次 `accelerator.save_model(model, "best_model_dir")`。

---

### 3. 你最关心的：模型参数是如何管理的？

在 `Accelerate` 模式下，参数管理有以下几个特点：

* **集中化：** 你所有的模型超参数（hidden_size, layers 等）都作为 `dict` 传给 `wandb`。这意味着你以后在 WandB 的网页上可以一眼看到：**“那个 mIoU=0.62 的实验，到底是用 (20, 8) 还是 (8, 8) 的 Patch 跑出来的？”** 这样就避免了用文件名记参数的混乱。
* **分布式安全：** `Accelerate` 确保只有主进程（Main Process）在写硬盘，避免了多卡同时写入导致的权重损坏。
* **动态监控：** 你甚至可以配置 `wandb.watch(model)`，它会每隔一段时间帮你记录模型各层梯度的直方图。如果梯度消失（GPR 深层信号学不到），你可以通过可视化第一时间发现。

### 总结
这套流程的**核心优势**在于：
1.  **极简：** 你的核心训练代码改动不超过 10 行。
2.  **严谨：** `TorchMetrics` 帮你搞定那四个论文指标，不用担心自己写的公式有 Bug。
3.  **可溯源：** `WandB` 帮你记住了每一次尝试。

**建议下一步：** 你可以先安装这几个库，尝试在你的 `run_verify_train` 脚本里加入 `accelerator.log`。一旦能从网页上看到那条 Loss 曲线动起来，你的“科研成就感”会瞬间爆棚！

## References

MCG GPR dataset: https://zenodo.org/records/14270869
Mamba-Net: https://arxiv.org/pdf/2601.17108
