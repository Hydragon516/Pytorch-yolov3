# Overview

Implement yolov3 with Pytorch. This code refers to https://github.com/ayooshkathuria/pytorch-yolo-v3.

# Requirements

Python 3.5, CUDA 9.0, Pytorch and other common packages listed in `requirements.txt`.

# Installation
### 1. Clone this repository

### 2. Install Pytorch
In anaconda environment, it can be installed with the following command.

```conda install -c peterjc123 pytorch cuda90```

### 3. Install dependencies
```pip install -r requirements.txt```

### 4. Download yolo.weights
Download data from https://pjreddie.com/media/files/yolov3.weights

# Sample test
### Run cam_demo.py
```python cam_demo.py```

# Result
![result.png](https://github.com/Hydragon516/Pytorch-yolov3/blob/master/images/result.png)
