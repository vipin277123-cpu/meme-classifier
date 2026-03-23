import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torchvision import transforms
from PIL import Image
from Model.CNN import MemeCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MemeCNN()
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meme_model.pt"))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

def predict_image(img_path):

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return "Vulgar"
    else:
        return "Non-Vulgar"