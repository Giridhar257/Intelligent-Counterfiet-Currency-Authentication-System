import torch
import cv2
import numpy as np
from PIL import Image
from models.multitask_model import MultiTaskModel
from utils.preprocessing import get_transforms
from utils.texture_features import extract_texture_features

device = torch.device("cpu")

model = MultiTaskModel().to(device)
model.load_state_dict(torch.load("cpu_model.pth", map_location=device))
model.eval()

denom_map = ["10", "20", "50", "100", "200", "500", "2000"]

def predict(image_path: str):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image not found or invalid path")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to PIL (VERY IMPORTANT)
    image_pil = Image.fromarray(image_rgb)

    texture = extract_texture_features(image)
    texture = torch.tensor(texture, dtype=torch.float32).unsqueeze(0)

    transform = get_transforms()
    image_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        auth_out, denom_out = model(image_tensor, texture)

    auth_pred = int(torch.argmax(auth_out, dim=1).item())
    denom_pred = int(torch.argmax(denom_out, dim=1).item())

    auth_label = "Real" if auth_pred == 0 else "Fake"

    return auth_label, denom_map[denom_pred]
