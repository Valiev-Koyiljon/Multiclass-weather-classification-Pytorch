
# Weather Condition Recognition with Deep Learning

![Weather Classification](link_to_image)

## Overview

This repository contains the code and documentation for a deep learning model designed to recognize weather conditions from single images. The model is trained on the "Multiclass Images for Weather Classification" dataset available on Kaggle. The project focuses on implementing a Convolutional Neural Network (CNN) using the PyTorch framework.

## Dataset

The dataset consists of 1125 labeled images, covering various weather conditions such as sunny, cloudy, rainy, snowy, or foggy. The dataset is divided into training and testing sets.

### Dataset Link
[Dataset on Kaggle](https://www.kaggle.com/datasets/somesh24/multiclass-images-for-weather-classification)

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Matplotlib
- Google Colab (for training in the provided notebook)

### Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/Multiclass-Weather-Classification-Pytorch.git
cd Multiclass-Weather-Classification-Pytorch
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preprocessing

1. Download the dataset from the provided [Google Drive link](https://drive.google.com/file/d/1sVJ4Y5zhMgj2dlyWHecdrMxNB3djisFT/view?usp=sharing).
2. Extract the dataset to your Google Drive.

## Model Architecture

The model architecture is based on ResNet9, a CNN architecture with residual blocks for enhanced performance. It includes various techniques such as normalization, data augmentation, and regularization to improve generalization.

## Training

To train the model, use the provided Colab notebook:
[Multiclass_weather_classification_CNN_Koyiljon.ipynb](https://colab.research.google.com/github/Valiev-Koyiljon/Multiclass-weather-classification-Pytorch/blob/main/Multiclass_weather_classification_CNN_Koyiljon.ipynb)

## Results

The model achieves an accuracy of over 92% on the validation set after 17 epochs.

### Training Time

The model was trained in approximately 3 minutes and 23 seconds.

## Testing

Test the model with individual images using the provided code in the notebook.

## Save the Model

The trained model weights are saved as `multiclass-weather-classification-resnet9.pth`. You can download it from [here](/content/multiclass-weather-classification-resnet9.pth).

## Conclusion

This project demonstrates the effectiveness of deep learning in weather classification tasks. The implementation includes advanced techniques to enhance model performance and prevent overfitting.

Feel free to explore and contribute to this project!

