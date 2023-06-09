# Setting up a Virtual Environment and Running a Python Program

1. Create a virtual environment using the following command:
-> conda create -n "pytorch_iris"


2. Install the necessary requirements or dependencies from the requirements.txt file using the command:
-> conda install -r requirements.txt


3. Activate the virtual environment by running the command:
-> conda activate pytorch_iris


4. Finally, execute your Python program located in the virtual environment using the command:
-> python iris.py


# Implementing a Neural Network in PyTorch using Iris Flower Dataset

-  Created synthetic dummy data using the torch.randn function as input features and random 
binary labels.
- Defined a simple neural network model using the nn.Sequential class, specifying the number of 
input and output neurons, hidden layers, and activation functions.
-  Utilized the Mean Squared Error (MSE) loss function and stochastic gradient descent (SGD) 
optimizer for training the model.
-  Trained the neural network for a specified number of epochs and recorded the loss values.
- Saved the trained model in two formats: the entire model and the model's state dictionary

                                  Loss vs Epoch Plot
![loss plot](https://github.com/aashir023/AI-Summer-Internship/assets/92915317/7ab704f7-9756-4330-b71c-f0d7a48bc41e)

                                    Confusion Matrix
![Confusion_matrix](https://github.com/aashir023/AI-Summer-Internship/assets/92915317/23acd88b-f7f4-4b06-ba8f-5efb9c1cff4a)

                                    
  
  
