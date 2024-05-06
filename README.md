
# Multi-Class Weather Classification with Deep Learning

## Overview
This project utilizes a convolutional neural network (CNN) model to classify images into one of four weather conditions: cloudy, rain, shine, or sunrise. The model is built using the PyTorch framework and trained on the "Multiclass Images for Weather Classification" dataset available on Kaggle.

## Dataset
The dataset contains 1125 labeled images divided into training and testing sets. Each image is labeled with one of the following weather conditions:
- Cloudy
- Rain
- Shine
- Sunrise

The dataset can be accessed [here](https://www.kaggle.com/datasets/somesh24/multiclass-images-for-weather-classification).

## Prerequisites
- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy

You can install the necessary libraries using the following command:
```bash
pip install torch torchvision numpy matplotlib
```

## Model Architecture
The model used is a variant of ResNet, specifically ResNet9, which includes layers like:
- Convolutional layers
- Batch normalization
- ReLU activations
- Max Pooling
- A fully connected layer

The architecture benefits from residual connections to facilitate deeper training without the model degrading.

## Training
The model was trained using:
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate Scheduling**: One-cycle policy with a maximum learning rate of 0.01
- **Regularization**: Weight decay and gradient clipping

Training involved both base image transformations (resizing and normalization) and augmentations (random cropping, horizontal flipping) to improve generalization.

## Results
The trained model achieved an accuracy of over 92% on the validation set. This high level of accuracy indicates that the model is well-generalized and performs well on unseen data.

## Files
- `train.py`: Script for training the model.
- `model.py`: Contains the CNN model architecture.
- `utils.py`: Helper functions for data loading and transformations.
- `evaluate.py`: Script for evaluating the model on the test set.

## Usage
To train the model, run:
```bash
python train.py
```
To evaluate the model on the test dataset, run:
```bash
python evaluate.py
```

## Visualization
We have included scripts that plot training/validation loss and accuracy which help in monitoring the training process. Additionally, the model's predictions on test images can be visualized to qualitatively assess its performance.

## Contributions
Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Distributed under the MIT License. See `LICENSE` for more information.

