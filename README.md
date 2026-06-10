# 在 MCG GPR 数据集上 复现 Mamba-Net

```
uv run accelerate launch -m src.train
uv run -m src.inference --files cc210 cc240 cc321
uv run -m src.inference --num_samples 5
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