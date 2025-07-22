FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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
    ca-certificates \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

# pip 최신화 + venv 사용 X
RUN python3 -m pip install --upgrade pip

RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

ENV PATH=/usr/local/cuda/bin:$PATH

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY docker_test.py docker_test.py

CMD [ "python", "docker_test.py" ]