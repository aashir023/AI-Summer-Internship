import pandas as pd
import torch
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
from sklearn import metrics


iris = load_iris()
df = pd.DataFrame(iris.data, columns= iris.feature_names)

df["target"] = iris.target

df["flower"] = df["target"].apply(lambda x : iris.target_names[x])

le = LabelEncoder()

df["flower"]= le.fit_transform(df["flower"])

X = df.drop(["target", "flower"], axis=1).values
Y = df["flower"].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state=42)

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)

y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

print(x_train.shape)
print(x_test.shape)
print(x_train.shape)
print(y_test.shape)

# class ANN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=4, out_features=16)
#         self.fc2 = nn.Linear(in_features=16, out_features=12)
#         self.output = nn.Linear(in_features=12, out_features= 3)

#     def forward(self, x):
#         x = f.relu(self.fc1(x))
#         x = f.relu(self.fc2(x))
#         x =self.output(x)
#         return X

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=4, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=12)
        self.output = nn.Linear(in_features=12, out_features=3)
        

    def forward(self, x):
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.output(x)
        return x


   
model = ANN()
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
      
epochs = 100
losses = []
for i in range(epochs):
    y_hat = model.forward(x_train)
    loss = criterion(y_hat, y_train)
    losses.append(loss.item())

    if i % 10 == 0:
        print(f"epoch {i} Loss { loss}")
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

preds = []

with torch.no_grad():
    for val in x_test:
     y_hat = model.forward(val)
     preds.append(y_hat.argmax().item())


from sklearn.metrics import classification_report

# Convert the true labels to string format
y_test_labels = [iris.target_names[label] for label in y_test]

# Convert the predicted labels to string format
y_pred_labels = [iris.target_names[pred] for pred in preds]

# Generate the classification report
report = classification_report(y_test_labels, y_pred_labels)

# Print the classification report
print(report)

data = pd.DataFrame({'y' : y_test, 'y_hat' :preds})

data["Correct"] = [1 if corr == pred else 0 for corr, pred in zip(data['y'], data['y_hat'])]
print(data.head())
print(data['Correct'].sum() / len(data))

import matplotlib.pyplot as plt
import seaborn as sns

class_names = ['versicolor', 'virginica', 'setosa']

confusion_matrix = metrics.confusion_matrix(y_test, preds)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("Confusion_matrix.png")
plt.show()
