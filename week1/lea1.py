import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class Model(nn.Module):
    def __init__(self,in_features=4,h1=8,h2=8,out_features=3):
        super().__init__()
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
torch.manual_seed(42)
model = Model()
#get data
import pandas as pd
import matplotlib.pyplot as plt
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'

my_df = pd.read_csv(url)
my_df['species'] = my_df['species'].map({'setosa':0,'versicolor':1,'virginica':2})
X = my_df.drop('species',axis=1).values
y = my_df['species'].values
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
train_X = torch.tensor(train_X,dtype=torch.float32)
test_X = torch.tensor(test_X,dtype = torch.float32)
train_y = torch.tensor(train_y, dtype=torch.long)
test_y = torch.tensor(test_y, dtype=torch.long)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#train model
epochs = 100
losses = []
for i in range(epochs):
    #go for a prediction
    y_pred = model.forward(train_X)
    #mearure loss
    loss = criterion(y_pred, train_y)
    #kepp track of Loss
    losses.append(loss.detach().numpy())

    # print every 10 epochs
    if(i % 10 == 0):
        pass
        #print("epoch:", i, "loss:", loss.item())
    #do some back propagation
    #through the network to adjust weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
#plot loss over time
#plt.plot(range(epochs),losses)
#plt.xlabel('epochs')
#plt.ylabel('loss')
#plt.show()
with torch.no_grad(): # no backpropagation needed
    y_val = model.forward(test_X)
    loss = criterion(y_val, test_y)
    #print("Validation loss:", loss.item())
    # # Predict iris species
    # sample = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).unsqueeze(0)
    # output = model.forward(sample)
    # prediction = torch.argmax(output,dim=1).item()
    # species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    # print("Prediction for [1.0,2.0,3.0,4.0]:", species_map[int(prediction)])
    correct = 0
    total = 0
    for i , data in enumerate(test_X):
        data = data.unsqueeze(0)  # Add batch dimension
        y_val = model.forward(data)
        pred = torch.argmax(y_val, dim=1).item()  # Get class index
        total += 1
        if(pred == test_y[i].item()):  # Compare with true label
            correct += 1      
    print("Accuracy:", correct / total)