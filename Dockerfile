FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Miniconda
RUN apt-get update && apt-get install -y wget git build-essential \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /opt/conda && rm /miniconda.sh
ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# Create environment
RUN conda create -n tree_detection python=3.9 -y

# Install CUDA toolkit
RUN source activate tree_detection && \
    conda install -c nvidia/label/cuda-12.1.0 cuda-toolkit -y

# Install pip and wheel
RUN source activate tree_detection && \
    pip install setuptools wheel

# Install PyTorch and CUDA 12.4 compatibility
RUN source activate tree_detection && \
    conda install -c pytorch \
        pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -y

# Install Detectron2
RUN source activate tree_detection && \
    conda install -c conda-forge detectron2 -y

# Copy code into container and install via setup.py
WORKDIR /app
COPY . .
RUN source activate tree_detection && \
    pip install .

# Use the environment by default
ENV PATH=/opt/conda/envs/tree_detection/bin:$PATH
ENV CONDA_DEFAULT_ENV=tree_detection

CMD ["python", "example/example.py"]