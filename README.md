# Subtle Defect Detection using YOLOv5 and Attention Mechanism

This project focuses on detecting defects in materials using the YOLOv5 objectdetection model enhanced with the Convolutional Block Attention Module (CBAM).
The models are trained to identify and classify subtle and camouflaged defects with high accuracy and efficiency.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction


## Features

- **High Accuracy Detection**: Utilizes the precision of YOLOv5 enhanced with the attention mechanism of CBAM.
- **Focus on Subtle Defects**: Specifically designed to identify subtle and easily missed defects.
- **Real-time Processing**: Optimized for quick processing to support real-time application scenarios.
- **Custom Synthetic Dataset**: Includes a dataset specifically created to train and test the model's effectiveness against targeted defect types.

## Prerequisites

Before you can run this project, you will need the following:
- Python 3.8 or higher
- PyTorch
- OpenCV
- NumPy
- Matplotlib

## Installation

To get started with this project, clone the repository and install the required dependencies:
Clone the current repository.
```bash
git clone https://github.com/yourusername/defect-detection-yolo.git
```

To run YOLOv5 model install the following dependencies:

cd ./code/yolov5
pip install -r requirements.txt


To run YOLOv8 model install the following dependencies:

cd ./code/ultralytics/
pip install -r requirements.txt


## Usage

To train an existing model on your custom/defect detection dataset:
- Go to the yolov5 directory
- create a dataset.yaml file in the same format for training yolov5
- Run the below script with your required parameters


```bash
python train.py --weights configs/yolov5_cbam.yaml --b
```
