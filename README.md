# Semantic Segmentation
## Introduction
The aim of this project is to detect the pixels of a road in images by using Fully Convolutional Network (FCN).


## Environment

The code has been developed and tested locally in a desktop computer with NVIDIA GPU. By the time of writing this, tensorflow-gpu latest version is 1.13, which requires cuDNN 7.5.1 and CUDA 9.2. 

### Requirements
- Download and install [NVidia Graphics Card Driver](https://www.nvidia.com/Download/index.aspx?lang=en-us)
- Download and install [CUDA Toolkit 9.2](https://developer.nvidia.com/cuda-92-download-archive?)
- Download and extract into CUDA installation folder [cuDNN 7.5.1](https://developer.nvidia.com/rdp/cudnn-download)
- Install [Anaconda3](https://www.anaconda.com/distribution/)
- Along with Anaconda3, there is the chance to install PyCharm IDE, which is the one that I decided to use for this project.

### Installation
- Clone or download this repository
- Create a new Anaconda Environment and install tensorflow or tensorflow-gpu
  *  You might be also required to install Pillow, tqdm and matplotlib
- Open the project with PyCharm and use the environment that you created in the step above as the interpreter

## Code

The project uses a [VGG16](https://machinelearningmastery.com/use-pre-trained-vgg-model-classify-objects-photographs/) convolutional neural network arquitecture that has been already pretrained for image recognition.

The hyperparameters has been tuned after performing several experiments until I got the results that I thought were acceptable:

* Epochs = 30 - I noticed that with 20 I also got good results and the loss more or less the same.
* Dropout rate (keep) = 0.6 - This reduces overfitting
* Learning rate = 0.001
* Batch size = 5

## Results

First of all, the loss graph shows how the traning reduces its loss in each epoch:
![](https://github.com/ferorga/CarND-Semantic-Segmentation/blob/master/results/loss_graph.png)

In the folder **resuluts** can be found all the test images that have been used to validate the NN:

![](https://github.com/ferorga/CarND-Semantic-Segmentation/blob/master/results/1556147355.008793/um_000000.png)
![](https://github.com/ferorga/CarND-Semantic-Segmentation/blob/master/results/1556147355.008793/um_000001.png)
![](https://github.com/ferorga/CarND-Semantic-Segmentation/blob/master/results/1556147355.008793/um_000002.png)
![](https://github.com/ferorga/CarND-Semantic-Segmentation/blob/master/results/1556147355.008793/um_000003.png)

## Conclusion and future work

With this project, it has been proved that the VGG16 NN works really good at clasifying the images. On the other hand, the semantic segmentation used to decode the output of the VGG16 seems to perform remarkably, even though I did not implemented any kind of image augmentation.
For future work, it might be useful to implement augmentation to avoid false positives that have been found in a few images.


