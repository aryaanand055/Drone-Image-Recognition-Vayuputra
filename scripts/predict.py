import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse

class_labels = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def load_model(path):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return class_labels[pred.item()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    model = load_model('models/eurosat_resnet50.pth')
    prediction = predict(args.image, model)
    print(f"Prediction: {prediction}")

if __name__ == '__main__':
    main()
