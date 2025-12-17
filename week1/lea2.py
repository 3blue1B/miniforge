from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Model definition
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=8, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Load trained model
torch.manual_seed(42)
model = Model()
model.load_state_dict(torch.load('../iris_model.pth', map_location='cpu'))
temp = torch.tensor([[1,2,3,4]], dtype=torch.float32)
temp = model.forward(temp)
rep = torch.argmax(temp, dim=1).item()
print(rep)

