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
Please see [our installation guide](installations.md) for installation instructions, either as docker-image or via conda.

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
