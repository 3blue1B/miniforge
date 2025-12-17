import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# convert image files into 4_dimension tensors
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./cnn_data',train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='./cnn_data',train=False,download=True,transform=transform)

# create smaller batches of data
train_loader = DataLoader(train_data,batch_size=100,shuffle=True)
test_loader = DataLoader(test_data,batch_size=100,shuffle=False)

#define the CNN model
#describe convolutional Layers
conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1)
conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,stride=1)

# #grab 1 MINST image 
# for i,(X_train,y_train) in enumerate (train_data):
#     break
# x = X_train.view(1,1,28,28)
# #print(x.shape)

# #first convolutional layer
# x = F.relu(conv1(x))
# #print(x.shape)

# #polling layer
# x = F.max_pool2d(x,kernel_size=2,stride=2)

# #second
# x = F.relu(conv2(x))
# x = F.max_pool2d(x,kernel_size=2,stride=2)

#model class
class ConVolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=3,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=3,stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=84)
        self.fc3 = nn.Linear(in_features=84,out_features=10)
    def forward(self,X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,kernel_size=2,stride=2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,kernel_size=2,stride=2)
        X = X.view(-1,16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

# create instance of this mode
torch.manual_seed(114514)
model = ConVolutionalNetwork()
# Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001) # the smaller the learing rate,the longer it take to train
import time
start_time = time.time()
#create variables to track them
epochs = 20
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# train
for i in range(epochs):
    trn_cor = 0
    tes_cor = 0
    train_loss = 0
    train_batches = 0
    model.train()
    for b,(X_train,y_train) in enumerate(train_loader):
        b += 1
        train_batches += 1
        y_pred = model(X_train)
        predicted = torch.max(y_pred.data,1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_cor += batch_corr
        loss = criterion(y_pred,y_train)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(b % 600 == 0):
            print(f'Epoch: {i} Batch: {b} loss: {loss.item()}')
    train_losses.append(train_loss / train_batches)
    train_correct.append(trn_cor)

#test
model.eval()
test_loss = 0
test_batches = 0
with torch.no_grad():
    for b,(X_test,y_test) in enumerate(test_loader):
        test_batches += 1
        y_val = model(X_test)
        predicted = torch.max(y_val.data,1)[1]
        batch_corr = (predicted == y_test).sum()
        tes_cor += batch_corr
        loss = criterion(y_val,y_test)
        test_loss += loss.item()
test_losses.append(test_loss / test_batches)
test_correct.append(tes_cor)

current_time = time.time()
total = current_time - start_time
print(f'Training took {total/60} minutes')
torch.save(model.state_dict(), '../mnist_cnn.pth')





