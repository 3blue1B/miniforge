import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import matplotlib.pyplot as plt  # For plotting graphs later

# Convert image files into 4D tensors
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./cnn_data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./cnn_data', train=False, download=True, transform=transform)

# Create smaller batches of data
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)
test_loader = DataLoader(test_data, batch_size=100, shuffle=False)

# Define the CNN Model Class
class ConvolutionalNetwork(nn.Module): # Minor typo fix in class name (optional)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, kernel_size=2, stride=2)
        X = X.view(-1, 16*5*5)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return X

# Create an instance of the model
torch.manual_seed(41)
model = ConvolutionalNetwork()

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --- Added: Learning Rate Scheduler ---
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

# --- Added: Variable to track the best model ---
best_test_loss = float('inf')

# Create variables to track metrics
epochs = 30
train_losses = []
test_losses = []
train_acc = []  # Will store training accuracy per epoch
test_acc = []   # Will store testing accuracy per epoch

start_time = time.time()

# Training Loop
for epoch in range(epochs):
    # ---- Training Phase ----
    model.train()
    trn_corr = 0  # Training correct predictions counter
    tst_corr = 0  # Testing correct predictions counter (for this epoch)
    running_train_loss = 0.0

    for batch_idx, (X_train, y_train) in enumerate(train_loader):
        # Forward pass
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        # Calculate accuracy for this batch
        predicted = torch.max(y_pred.data, 1)[1]
        batch_corr = (predicted == y_train).sum()
        trn_corr += batch_corr

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    # Calculate average training loss and accuracy for the epoch
    avg_train_loss = running_train_loss / len(train_loader)
    epoch_train_acc = trn_corr.item() / len(train_data) * 100  # Accuracy in %
    train_losses.append(avg_train_loss)
    train_acc.append(epoch_train_acc)

    # ---- Validation/Testing Phase (Each Epoch) ----
    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for X_test, y_test in test_loader:
            y_val = model(X_test)
            loss = criterion(y_val, y_test)
            running_test_loss += loss.item()

            predicted = torch.max(y_val.data, 1)[1]
            batch_corr = (predicted == y_test).sum()
            tst_corr += batch_corr

    # Calculate average test loss and accuracy for the epoch
    avg_test_loss = running_test_loss / len(test_loader)
    epoch_test_acc = tst_corr.item() / len(test_data) * 100  # Accuracy in %
    test_losses.append(avg_test_loss)
    test_acc.append(epoch_test_acc)

    # Update learning rate based on test loss
    scheduler.step(avg_test_loss)

    # Save the model if it's the best so far
    if avg_test_loss < best_test_loss:
        best_test_loss = avg_test_loss
        torch.save(model.state_dict(), 'mnist_cnn.pth')
        print(f'Epoch {epoch:2d}: ** New best model saved! (Test Loss: {avg_test_loss:.4f})')

    # Print epoch summary
    print(f'Epoch: {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {epoch_train_acc:6.2f}% | Test Loss: {avg_test_loss:.4f} | Test Acc: {epoch_test_acc:6.2f}%')

total_time = time.time() - start_time
print(f'\nTraining complete. Total time: {total_time/60:.2f} minutes')
print(f'Final Test Accuracy: {test_acc[-1]:.2f}%')

# Save the final model state
#torch.save(model.state_dict(), 'mnist_cnn.pth')
print("Model saved as 'mnist_cnn.pth'")

# ----- (Optional) Plotting Training History -----
plt.figure(figsize=(12, 4))
# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_acc, label='Training Accuracy')
plt.plot(test_acc, label='Test Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.tight_layout()
plt.show()