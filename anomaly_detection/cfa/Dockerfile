FROM python:3.11-slim

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    gcc \
    g++ \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libproj-dev \
    libsndfile1 \
    libssl-dev \
    vim \
    unzip \
    zip \
    tmux \
    wget \
 && apt-get clean && rm -rf /var/lib/apt/lists/*


# pip 최신화
RUN pip install --upgrade pip

# 주요 패키지 설치
RUN pip install \
    jupyterlab ipywidgets yapf tqdm \
    matplotlib scikit-learn scikit-image \
    torch==2.2.2 torchvision==0.15.2 \
    tensorflow-cpu==2.13.0 keras tensorflow-hub \
    mxnet==1.9.1 \
    onnx onnxruntime-gpu optimum[onnxruntime-gpu] \
    pycocotools \
    opencv-python \
    einops \
    kornia \
    pillow \
    addict \
    termcolor \
    cython \
    submitit \
    mlflow \
    vessl


# /requirements.txt 를 복사해 추가 설치
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.txt \
 && rm -rf /root/.cache /tmp/*.whl


# 작업 디렉토리 설정
WORKDIR /workspace
COPY anomaly_detection/cfa .

CMD ["bash"]