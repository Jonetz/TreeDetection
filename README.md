# TreeDetection
Python Software for countrywide detection and delineation of tree crowns based on a trained ResNet Model. Developed by [Luca Reichmann](https://github.com/reichmla), [Jonas Nasimzada](https://github.com/JonasNasimzada), [Alina Roitberg](https://aroitberg.github.io/), and [Jonas Geiselhart](https://github.com/Jonetz) at the University of Stuttgart in Corporation with the [Office of Geoinformation and Land-development Baden-Württemberg](https://www.lgl-bw.de/).

## Usage
In order to infer custom images, a config with all parameters must be constructed and a function can be called to execute inference based on all parameters in the config. 

The config can either be given as a YAML file or it can be hardcoded as directory. Here the input directories, models, and filename format. Please refer to the YAML in the reposito and the get config method to see all tuneable parameters. 

The primary functions to be used are in the main file, the process_files method is designed self contained and works only with the given parameters, if you choose to rely on the single functions to have more flexibility, you can also use preprocess_files, predict_tiles, and postprocess_files. In Order for this to work, the corresponding data from the previous steps should be made available before calling any of these methods.

The program can be executed with either one or two models based on whats given in the config, if two models are choosen also a segmentation boundary as shape needs to be provided. 

## Additional Data 
Additional data such as models [(here)](https://drive.google.com/drive/folders/1kc9LbImT5-2bl0A86FfWho2q9qx7MBYd?usp=drive_link), example images and height maps (in the data folder) and training datasets can be found here in the future.

## Installation
Typically the installation of [Detectron](detectron2.readthedocs.io/en/latest/tutorials/instal), [CUDA](https://developer.nvidia.com/cuda-12-1-0-download-archive) and [GDAL](https://gdal.org/en/stable/) runs into combatibility issues, we provide a way to install it using Conda (which is strongly adviced) and Cuda 12.1.
It is also possible to run on other versions such as CUDA 11.3 or CUDA 12.5, for this besides the versions here, the exact cupy version in the setup.py has to be adapted.


### Using Conda

1. Clean the cache:
   ```bash
   pip cache purge
   ```

2. Create a new Environment:
   ```bash
   conda create -n tree_detection python=3.9
   ```

3. Install the Nvidia Cuda Toolkit for GPU support, we suggest using cuda12x:
   ```bash
   conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0
   ```

4. Install PyTorch via an alternative channel:
   ```bash
   pip install setuptools wheel
   conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   ```

5. Install Detectron via Conda:
   ```bash
   conda install -c conda-forge detectron2
   ```

6. Install the package:
   ```bash
   pip install git+https://www.github.com/Jonetz/TreeDetection/
   ```


### Using Docker
The Dockerfile is provided for installation of a container. Currently a development version has to be installed, which is quite large (~ 37 GB) and can take up to half an hour to install.
First ensure you have installed CUDA drivers compatible with CUDA 12.6 on your PC. Ypu can check this by runnning nvidia-smi in the command prompt and view the current driver version.
In order to install the software now, you can just downlad the repository, move in the directory and build & run the container as follows:
   ```bash
   docker build -t tree-detection .
   docker run --gpus all -it -v %cd%:/tree-detection tree-detection
   ```
After building the container you need to load the pretrained weights/models and Forest-Shapes as well as other Exclude-Shapes in the data directory:
   ```bash
   docker cp "< Your Models Path >" tree-detection:/app/data/
   ```
At the end you can test the installation with running the example script in a docker terminal:
   ```bash
   python example/example.py
   ```
### Common Problems
- AssertionError: Input path is missing from the configuration or path is incorrect. -> Indicates, you run the example program from the wrong directory, use the toplevel directory to run the example.
- AssertionError: < Type > model path is missing from the configuration or path is incorrect. -> Indicates, that the proper models where not imported into the right directory or the config
- AssertionError: Forrest outline path is missing from the configuration.
- cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version -> Indicates that the program is unable to reach the GPU, check the driver access first on your host machine and second in your container, search for missing access or driver incompatibilities.
- missing .h or .cpp Files -> Typically also a driver mismatch, try reloading the build essentials, the cuda toolkit or cupy 

### Requirements
The program requires a GPU with an adequate installed CUDA version for training and inference.
Further, the following libraries are used:

```python
affine
aiofiles
cupy_cuda113 / cupy-cuda12x / ... (depending on your cuda version)
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


## Training and Inference 

Illustration of the different steps during training and inference, that can be applied through the framework.
<!-- ![Training description](supplementary/imgs/illustration1.jpg) -->
<img src="supplementary/imgs/illustration1.jpg" width="50%" />

## Supplementary Data 
We provide supplementary data for training the models, segmentation of the box annotations to more fine grained annotations, generation of autolabels, and model evaluation.

### Visual Samples

Sample in Baden-Württemberg, Southern Germany             |  Sample of the University of Stuttgart
:-------------------------:|:-------------------------:
![Sample in Baden-Württemberg, Southern Germany](supplementary/imgs/sample1.jpg)  |  ![Sample of the University of Stuttgart](supplementary/imgs/sample2.jpg)

Sample of Stuttgart downtown             |  Sample of a forest near the University
:-------------------------:|:-------------------------:
![Sample of Stuttgart downtown](supplementary/imgs/sample3.jpg)  |  ![Sample of a forest near the University](supplementary/imgs/sample4.jpg)


Illustration of our autolabel generation using height maps, as given in the supplementary material 
<!-- ![Autolabel Generation](supplementary/imgs/illustration2.jpg) -->
<img src="supplementary/imgs/illustration2.jpg" width="50%" />
