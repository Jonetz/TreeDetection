FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# --- [1] Install system-level dependencies for OpenCV and other packages ---
RUN apt-get update && apt-get install -y \
    wget git build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda && rm /miniconda.sh

# --- [2] Add conda to path ---
ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# --- [3] Create conda environment and install PyTorch + Detectron2 ---
RUN conda create -n tree_detection python=3.9 -y && \
    conda run -n tree_detection conda install -c pytorch \
        pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -y && \
    conda run -n tree_detection pip install setuptools wheel && \
    conda run -n tree_detection pip install 'git+https://github.com/facebookresearch/detectron2.git'

# --- [4] Copy your codebase into the container ---
WORKDIR /app
COPY . .

# --- [5] Install your Python package into the environment ---
RUN conda run -n tree_detection pip install .

# --- [6] Set environment for container sessions ---
ENV PATH=/opt/conda/envs/tree_detection/bin:$PATH
ENV CONDA_DEFAULT_ENV=tree_detection

# --- [7] Show message when container is run ---
CMD ["bash", "-c", "echo Container started.; exec bash"]