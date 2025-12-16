from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

app = Flask(__name__)

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
model_path = os.path.join(os.path.dirname(__file__), 'iris_model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            sample = torch.tensor([[sepal_length, sepal_width, petal_length, petal_width]], dtype=torch.float32)
            output = model(sample)
            prediction = torch.argmax(output, dim=1).item()
        
        species = species_map[prediction]
        return render_template('index.html', 
                             prediction=species,
                             sepal_length=sepal_length,
                             sepal_width=sepal_width,
                             petal_length=petal_length,
                             petal_width=petal_width)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
