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


## References

MCG GPR dataset: https://zenodo.org/records/14270869
Mamba-Net: https://arxiv.org/pdf/2601.17108


https://mp.weixin.qq.com/s?__biz=Mzg4MzU5OTg5Nw==&mid=2247492581&idx=1&sn=b12f94474adc189fb135f1e156bf4594&chksm=cee4ff2892e5a7bb2766e8c2638c1eb3951673cb1b05c05c119b8e3d81901899553a0dc56212&mpshare=1&srcid=04235hFoTrhHRJm2NmokH18r&sharer_shareinfo=7c7530eba32f389158d0d576d3bc9352&sharer_shareinfo_first=7c7530eba32f389158d0d576d3bc9352&from=timeline&scene=2&subscene=1&sessionid=1776987229&clicktime=1776987920&enterid=1776987920&ascene=2&fasttmpl_type=0&fasttmpl_fullversion=8227329-zh_CN-zip&fasttmpl_flag=0&realreporttime=1776987920089&devicetype=android-36&version=28003933&nettype=WIFI&lang=zh_HK&exportkey=n_ChQIAhIQ8YvOMC3miuMz5J08J8V58hL2AQIE97dBBAEAAAAAAAr6IewMRRUAAAAOpnltbLcz9gKNyK89dVj0%2F8ORjFyPB9rQQBT%2Fh6RDSgOijjuP%2BCLrZ6Vm1mMSwv9dj0uloBCjuG0Dbnr4NTkW84MZ79KRg0C6L6%2BUPo0e2eJTkl%2FyKAYVxCYKO8NREzvuGBlQOC5uedTzd9w8VE%2BLst7EwGTkAP9SsqP5jmmrsj%2BAgw34cpGQ%2Fe8P6DIw0uAVfodTsQU%2F%2BQQoYwGsS7jQYwUMjlYJGsD%2FuBygvcRIQ7Z9q8XUEfTjCDeF3UFzweFex6rzcFZSqvvwgvaMtCFP6DDKME46TVv4yI4JG76F%2Bw%3D%3D&pass_ticket=yx7O80LPpyVMuwUucPElN%2BY3uINsR9eeFbpR1ZFSYaaaR9f1aNSw99YSxS1GzWIP&wx_header=3