# Setting up a Virtual Environment and Running a Python Program

1. Created a virtual environment using the following command:
 conda create -n "pytorch_nn"


2. Installed the necessary requirements or dependencies from the requirements.txt file using the command:
 conda install -r requirements.txt


3. Activated the virtual environment by running the command:
 conda activate pytorch_nn


4. Finally, execute my Python program located in the virtual environment using the command:
 python torch_nn.py


# Implementing a Neural Network with PyTorch using Dummy Data
-  Created synthetic dummy data using the torch.randn function as input features and random 
binary labels.
- Defined a simple neural network model using the nn.Sequential class, specifying the number of 
input and output neurons, hidden layers, and activation functions.
-  Utilized the Mean Squared Error (MSE) loss function and stochastic gradient descent (SGD) 
optimizer for training the model.
-  Trained the neural network for a specified number of epochs and recorded the loss values.
-  Saved the trained model in two formats: the entire model and the model's state dictionary

                                            Loss vs Epochs PLot
                                            
                                        