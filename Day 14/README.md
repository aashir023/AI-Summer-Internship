# CIFAR-10 Image Classification using Convolutional Neural Networks

This code demonstrates how to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Dataset

The CIFAR-10 dataset is used for training and testing the model. It is loaded using the torchvision library and preprocessed as follows:
- Images are resized to `(32, 32)` pixels.
- Images are converted to tensors.
- Pixel values are normalized to have a mean of `[0.4914, 0.4822, 0.4465]` and a standard deviation of `[0.2023, 0.1994, 0.2010]`.

## Model Architecture

The CNN model used in this code consists of multiple convolutional layers, followed by max pooling and fully connected layers. The architecture can be summarized as follows:

- Convolutional Layer 1: `in_channels=3`, `out_channels=32`, `kernel_size=3`
- Convolutional Layer 2: `in_channels=32`, `out_channels=32`, `kernel_size=3`
- Max Pooling Layer 1: `kernel_size=2`, `stride=2`
- Convolutional Layer 3: `in_channels=32`, `out_channels=64`, `kernel_size=3`
- Convolutional Layer 4: `in_channels=64`, `out_channels=64`, `kernel_size=3`
- Max Pooling Layer 2: `kernel_size=2`, `stride=2`
- Fully Connected Layer 1: `input_size=1600`, `output_size=128`
- ReLU Activation
- Fully Connected Layer 2: `input_size=128`, `output_size=num_classes`

The model takes images as input with a shape of `(batch_size, 3, 32, 32)` where `3` represents the number of channels (RGB) and `32x32` represents the image size. The output of the model is a probability distribution over the `num_classes` classes.

## Training

The model is trained using Stochastic Gradient Descent (SGD) optimizer with a learning rate of `0.001`, weight decay of `0.005`, and momentum of `0.9`. The loss function used is Cross Entropy Loss.

The training is performed for `num_epochs=20` with a batch size of `64`. The training loop iterates over the training data and performs forward pass, backward pass, and optimization in each iteration. The loss is printed after each epoch.

## Evaluation

After training, the accuracy of the trained model is evaluated on the training set. The accuracy is calculated by comparing the predicted labels with the true labels.

## Usage

To use this code, make sure you have the necessary dependencies installed. You can run the code in a Python environment with PyTorch and torchvision libraries installed.

Feel free to modify the code according to your requirements and dataset.

