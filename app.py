from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import io
import base64
import numpy as np
import torchvision.transforms as transforms
import re

app = Flask(__name__)

# Define the CNN model class (same as in your model.py)
class ConVolutionalNetwork(nn.Module):
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

# Load the trained model
def load_model():
    model = ConVolutionalNetwork()
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Image preprocessing function
def preprocess_image(image_data):
    # Remove data URL prefix if present
    if 'base64,' in image_data:
        image_data = image_data.split('base64,')[1]
    
    # Decode base64 image
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
    
    # Invert colors (MNIST has white digits on black background)
    image = Image.eval(image, lambda x: 255 - x)
    
    # Resize to 28x28
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Normalize to [0, 1] range
    img_array = img_array / 255.0
    
    # Convert to tensor with batch dimension
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Add batch dimension
    tensor = transform(img_array).unsqueeze(0)
    tensor = tensor.to(torch.float32)
    return tensor

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image']
        
        # Preprocess the image
        input_tensor = preprocess_image(image_data)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            
        # Format the response
        result = {
            'prediction': int(prediction.item()),
            'confidence': float(confidence.item() * 100),
            'probabilities': [float(p) * 100 for p in probabilities[0].numpy()]
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/clear', methods=['POST'])
def clear():
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)