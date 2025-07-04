# Installation Guide
Typically the installation of [Detectron](detectron2.readthedocs.io/en/latest/tutorials/instal), [CUDA](https://developer.nvidia.com/cuda-12-1-0-download-archive) and [GDAL](https://gdal.org/en/stable/) runs into combatibility issues.

## 1. Using Conda

### Download the Repository
In order to install the Package & run the sample script, it is best to download the repository and work within the repository:
   ```bash
   git clone https://github.com/Jonetz/TreeDetection
   cd Tree Detection
   ```

### Ensuring GPU-Drivers are installed
First ensure you have installed CUDA drivers compatible with CUDA 12.6 on your PC. You can test this by running
   ```bash
   nvidia-smi
   ```
to receive  an overview of your drivers version, the current CUDA Version and your GPUs together with current resources allocations.
Then test the current Nvidia Cuda Compiler version by checking nvcc 
   ```bash
   nvcc --version
   ```
this should also return a compatible CUDA Version.  If this does not return any versions you can install the compilers via:
   ```bash
   sudo apt-get install cuda-toolkit
   ```
Afterwards try the nvcc-version again. 

Install Build-Essentials, to build the Detectron application:
   ```bash
   sudo apt-get install build-essential
   ```
Try to verify that gcc & g++ work by trying:
   ```bash
   gcc --version
   ```
If no version is shown try to install gcc & g++ via apt.


It is important to note the CUDA Version here if it starts with 12.X you do not have to worry otherwise you should either consider updating/downgrading your version if possible, or you need to adapt the versions in the next steps suitably. 

If you do not already have the right drivers installed please refer to the installation guides:
- [Installation Guide using Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Installation Guide using Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
- [Installation Guide using WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

For those interested in the technical details of CUDA compatibility, see the [official Nvidia compatibility website](https://docs.nvidia.com/deploy/cuda-compatibility/index.html).

### Ensure Conda is installed
You can check if conda is already installed by running:
   ```bash
   conda --version
   ```
which should output something like this (the exact version number may vary):
   ```bash
   conda 25.3.1
   ```
If the command is not recognized, you can download and install Miniconda from [the conda website](https://docs.conda.io/en/latest/miniconda.html). Please follow the instructions for your OS and check again if conda is installed in the end.

### Create your Environment
With this command create your conda environment named tree_detection that has the right versions of python, torch, torchvisions, etc. already installed. You can again execute this in the topdirectory of the repository:
   ```bash
   conda env create -f environment.yml
   conda activate TreeDetection
   ``` 
If you have another version than CUDA 12.X you will most likely run into incompatibilites, in this case look at the versions in the environment file / requirements and install them manually.
The torch compatibility charts might help to find suitable versions: [pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). Here it is advantageous to already look ahead and compare the compatible versions in the next step.

### Install the Nvidia Toolkit
Use conda to install the NVIDIA CUDA Toolkit:
   ```bash
   conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0
   ``` 
If this fails or you already know that your PC is not compatible with CUDA 12 you can search for a good version on the [conda reference for cuda toolkits](https://anaconda.org/nvidia/cuda-toolkit).

### Check if the installation worked
Now the sample can be tried by copying the model to the data directory and start the prediction:
   ```bash
   cp <your modelpath> data/model_combined.pth
   python example/example.py
   ``` 
This should now predict the first sample image and write the result in the output folder.
### Troubleshooting and manual installations 
Detect if something went wrong, by checking the actual code availability of all packages:
   ```bash
   # PyTorch version
   python -c "import torch; print('PyTorch version:', torch.__version__)"
   # TorchVision version
   python -c "import torchvision; print('TorchVision version:', torchvision.__version__)"
   # CUDA availability and version via PyTorch
   python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version (PyTorch):', torch.version.cuda)"
   # cuDNN version
   python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version())"
   # GPU name (if CUDA is available)
   python -c "import torch; print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
   # Detectron2 version
   python -c "import detectron2; print('Detectron2 version:', detectron2.__version__)"
   # CuPy version
   python -c "import cupy; print('CuPy version:', cupy.__version__)"
   # CuPy CUDA availability and GPU info
   python -c "import cupy; dev = cupy.cuda.Device(); print(f'CuPy CUDA device ID: {dev.id}')"
   # System-level CUDA toolkit version (via nvcc)
   nvcc --version
   ```
Which should result in something like this, the exact versions may differ: 
   ```bash
   PyTorch version: 2.5.1
   TorchVision version: 0.20.1
   CUDA available: True
   CUDA version (PyTorch): 12.4
   cuDNN version: 90100
   Device name: NVIDIA GeForce RTX 4090
   Detectron2 version: 0.6
   CuPy version: 13.4.1
   CuPy CUDA device ID: 0
   nvcc: NVIDIA (R) Cuda compiler driver
   Copyright (c) 2005-2023 NVIDIA Corporation
   Built on Fri_Jan__6_16:45:21_PST_2023
   Cuda compilation tools, release 12.0, V12.0.140
   Build cuda_12.0.r12.0/compiler.32267302_0
   ```
If there are some problems or the CUDA Version is not compatible with the given package options, try manually installing the missing packages:

1.) Install Detectron2 
- Facebooks [Detectron2 libary](https://github.com/facebookresearch/detectron2) needs to be compiled specifically for your CUDA version, so it must be installed separately, under windows this requires several additional libaries, so linux is recommended:
   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ``` 
   If this returns a error you can try installation via conda using the conda-forge channel  `conda install -c conda-forge detectron2`, ensure version 0.6 was installed.
- Again if you have a different CUDA or Torch versions please refer to the [official installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to find the correct version. 
- You can check if the installation was successful using this command, which should return a version number:
   ```bash
   python -c "import detectron2; print(detectron2.__version__)"
   ``` 
2.) Install Torch and torchvision
- Try using alternative channels:
   ```bash
   pip install setuptools wheel
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   ``` 

3.) Install TreeDetection
- After installing all other packages, it should be possible to install the TreeDetection library using:
   ```bash
   pip install git+https://www.github.com/Jonetz/TreeDetection/
   ``` 

## 2. Using Docker
The Dockerfile is provided for installation of a container. Currently a development version has to be installed, which is quite large and can take up to half an hour to install.

### Ensuring GPU-Drivers are installed
First ensure you have installed CUDA drivers compatible with CUDA 12.6 on your PC. You can test this by running
   ```bash
   nvidia-smi
   ```
to receive an overview of your drivers version, the current CUDA Version and your GPUs together with current resources allocations.
Then test the current Nvidia Cuda Compiler version by checking nvcc 
   ```bash
   nvcc --version
   ```
this should also return a compatible CUDA Version. 

### Install via Docker
In order to install the software now, you can just downlad the repository, either via using the git website download or via git using
   ```bash
   git clone https://github.com/Jonetz/TreeDetection
   ```
afterwards move in the top-directory and build the container as follows:
   ```bash
   docker build -t tree-detection .
   ```
Then you can check wheter the container image was succesfully installed via:
   ```bash
      docker image ls
   ```
which should output something like this
   ```bash
   REPOSITORY       TAG       IMAGE ID       CREATED          SIZE
   tree-detection   latest    XXXXXXXXXXX   XY minutes ago   32.5GB
   ```
Then you can run this as in image either with no mounted volume 
   ```bash
   docker run --gpus all -it --name tree-detection tree-detection bash
   ```
or if you want it to be mounted to a volume:
   ```bash
   docker run --gpus all -it -v $(pwd):/tree-detection tree-detection bash # In linux 
   docker run --gpus all -it -v %cd%:/tree-detection tree-detection # In windows
   ```
After building the container you need to load the pretrained weights/models and Forest-Shapes as well as other Exclude-Shapes in the data directory:
   ```bash
   docker cp "Your data/path" tree-detection:/app/data/
   ```
If the copy command returns a folder not found error this could be due to the container not being started or being assigned another name due to muliple starts. Check your current running containers in this case!

At the end you can test the installation with running the example script in a docker bash terminal:
   ```bash
   python example/example.py
   ```
## Common Problems
1. AssertionError: Input path is missing from the configuration or path is incorrect. 
→ Indicates, you run the example program from the wrong directory, use the toplevel directory to run the example.
2. AssertionError: < Type > model path is missing from the configuration or path is incorrect.
→ Indicates, that the proper models where not imported into the right directory or the config
3. AssertionError: Forrest outline path is missing from the configuration.
→ Indicates, that the cofnig is using separate models based on forest environments, and the forest shape is missing
4. RuntimeError: operator torchvision::nms does not exist
→ Indicates, that the installed torch and torchvision versions are incompatible, try using another torchvision by reinstalling it without caching.
5. cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version 
→ Indicates that the program is unable to reach the GPU, check the driver access first on your host machine and second in your container, search for missing access or driver incompatibilities.
6. Missing .h or .cpp Files 
→ Typically also a driver mismatch, try reloading the build essentials, the cuda toolkit or cupy 

## Requirements
The program requires a GPU with an adequate installed CUDA Version for training and inference.
Further, the following libraries are used:

```python
affine
aiofiles
cupy_cuda113 / cupy-cuda12x / ... (depending on your CUDA Version)
detectron2
fiona
geopandas
matplotlib
numba
numpy
opencv_python
pandas
pycocotools
PyYAML
rasterio
scikit_learn
scipy
Shapely
skimage
torch
```
Typically, `rasterio` and `geopandas` require some form of [GDAL](https://gdal.org/en/stable/), which comes in many versions and often conflicts with `torch` or `detectron`, so watch out for this during installation. 
Additionally, `detectree` in version 1.0.8 is needed for training. This can be downloaded [here](https://github.com/PatBall1/detectree2/releases), for this it is advised to use CUDA 11.3.

