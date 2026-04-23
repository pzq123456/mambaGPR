# 在 MCG GPR 数据集上 复现 Mamba-Net\


gpr max


## References

MCG GPR dataset: https://zenodo.org/records/14270869
Mamba-Net: https://arxiv.org/pdf/2601.17108


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

### 3. 一个致命细节提醒

请务必检查一下：**`annotations` 下的文件名是否与 `images` 下的文件名完全一致？**
从你的扫描看：
* `images/D_TRAIN` 有 `AA1.png`
* `annotations/D_TRAIN` 也有 `AA1.png`
这非常好！这意味着它们是 **Pixel-level Mask**。

**但是！** 检查一下 `AA1.png` 在 `annotations` 里的像素值。
* 如果是 **0 和 255**（黑白图），你需要显式地在 DataLoader 里把它们除以 255 变成 **0 和 1**。
* 如果是 **0 和 1**，肉眼看过去是全黑的，别以为文件坏了，那是正常的分类标签。


## 训练配置建议
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: 'NO'
downcast_bf16: 'no'
enable_cpu_affinity: true
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```