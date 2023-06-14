# Setting up a Virtual Environment and Running a Python Program

1. Create a virtual environment using the following command:
 conda create -n "numpy_nn"


2. Install the necessary requirements or dependencies from the requirements.txt file using the command:
 conda install -r requirements.txt


3. Activate the virtual environment by running the command:
 conda activate numpy_nn


4. Finally, execute your Python program located in the virtual environment using the command:
 python nn.py


# Implementing a Neural Network with Numpy on the Cat vs. Non-Cat Dataset

- Loaded the Cat vs. Non-Cat dataset using the h5py library.
- Processed and prepared the dataset by reshaping the input features and standardizing the data.
- Defined a neural network architecture with an input layer, hidden layer, and output layer using the sigmoid activation function.
- Implemented the gradient descent algorithm to optimize the network parameters and minimize the cost function.
- Evaluated the trained model's performance by computing the accuracy on both the train and test sets.
