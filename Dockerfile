# 1. 使用 NVIDIA 官方提供的带有 CUDA 开发工具的镜像 (这是避开“配环境”的核心)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 2. 安装 GCC-12 和基础依赖
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    gcc-12 \
    g++-12 \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 3. 设置环境变量，模拟你在文档中手动 export 的内容
ENV PATH=/usr/local/cuda-12.8/bin:$PATH
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# 关键点：指定编译器
ENV CC=gcc-12
ENV CXX=g++-12
# 针对你的显卡架构，这里建议设为 "8.9" (L40) 或 "8.0" (A100)，"10.0"
ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV MAMBA_FORCE_BUILD=TRUE

WORKDIR /app

# 4. 迁移你的 Python 环境 (假设你使用 uv)
RUN pip install uv
COPY . .
RUN uv venv
RUN source .venv/bin/activate && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install causal-conv1d --no-cache && \
    uv pip install mamba-ssm --no-cache

CMD ["/bin/bash"]