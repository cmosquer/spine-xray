# Landmark detection in lateral spine x-rays

The evaluation of spinal alignment with radiological parameters is essential in patients with spinal conditions likely to be treated surgically.
This preoperative information may determine the need for sagittal balance corrections when planning surgery or preventing iatrogenic spinal deformities, 
even as the result of short fusions. Although these parameters are measured in a whole-spine radiograph (WS x-rays), they are not usually included in the 
radiological report, so the measurement is commonly performed by spinal surgeons, which is time-consuming and subject to errors. 


This module applies convolutional networks to detect the coordinates of 31 keypoints in a lateral spine x-ray.
The dataset used for training should be labeled with COCO annotations. The module features WandB for experiment monitoring and configuration.
This includes launching WandB sweeps to optimize hyperparameters: 
![image](https://user-images.githubusercontent.com/43885984/211001983-da4a40cc-1ff7-423e-be54-946839efefab.png)

* Architecture: U-Net or FPN (Feature pyramid network) 
* Backbones: ResNet50 // Densenet 121//  Efficient Net B0//  Efficient Net B4 // Efficient Net B5 // xception // Dual path network 68
* Levels
* Optimizer
* Loss function
* Learning rate
* Batch size
* Data Augmentation
* Epochs
* Sigma
* Patience for early stopping

The configuration of values for sweeps is done in YAML files in configs/sweep, and file reference is specified in main.py

There are two main pipelines implemented: 

* Direct Coordinates Regression (DCR)
* Heatmap-based Regression (HBR)

 
