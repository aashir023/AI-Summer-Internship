First I created a virtual environment using:
-> conda create -n "env_name"

Then I installed the requirements.txt file which contains the necessary requirements or dependancies needed to run the program succesfully by using
-> conda install -r requirements.txt

After installing the requirements.txt file I activated the virtual environment i just created using:
-> conda activate env_name

Finally I executed my python file located in the virtual environment using:
-> python filename.py



Implementing a Neural Network with Numpy on the Cat vs. Non-Cat 
Dataset
 Loaded the Cat vs. Non-Cat dataset using the h5py library.
 Processed and prepared the dataset by reshaping the input features and standardizing the data.
 Defined a neural network architecture with an input layer, hidden layer, and output layer using 
the sigmoid activation function.
 Implemented the gradient descent algorithm to optimize the network parameters and minimize 
the cost function.
 Evaluated the trained model's performance by computing the accuracy on both the train and test 
sets.