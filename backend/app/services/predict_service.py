import torch
from PIL import Image
from ..core.config import LABELS
from ..core.preprocessing import eval_transform
from .model_service import load_model, DEVICE


def predict(image: Image.Image, model_name: str) -> dict:
    model = load_model(model_name)

    tensor = eval_transform(image.convert("L")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    predicted_index = int(probs.argmax())

    return {
        "label":             LABELS[predicted_index],
        "confidence":        round(float(probs.max()) * 100, 2),
        "normal_prob":       round(float(probs[0]) * 100, 2),
        "pneumonia_prob":    round(float(probs[1]) * 100, 2),
        "model_name":        model_name,
    }
