from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
import torch
import torchvision.transforms as transforms
from model import VGG
import torch.nn as nn
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor()
])

app = Flask(__name__)
model_path = '../Pizza_or_not/vgg.pth'
model = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    global model, device
    model = VGG()
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    global model, device
    model.to(device)
    file = request.files['file']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes))
    image = transform(image).unsqueeze(0)
    image.to(device)
    with torch.no_grad():
        output = model(image.to(device))
    pred = output.argmax().item()
    return jsonify({'prediction': pred})


load_model()
app.run(debug=False)