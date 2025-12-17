import torch
import torch.nn as nn
import torch.optim as optim

# synthetic data: y = 3x + 2
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = 3 * x + 2 + 0.2 * torch.randn(x.size())

# simple linear model using nn.Module
model = nn.Linear(1, 1)
criterion = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(200):
    opt.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    opt.step()
print("params:", list(model.parameters()))
