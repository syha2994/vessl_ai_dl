FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-opencv \
    git \
    wget \
    build-essential \
    ninja-build \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# pip 최신화 + venv 사용 X
RUN python3 -m pip install --upgrade pip

# PyTorch (CUDA 12.1, pip 기반)
RUN pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

ENV PATH=/usr/local/cuda/bin:$PATH

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY docker_test.py docker_test.py

CMD [ "python", "docker_test.py" ]