import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from ..core.config import MODELS_DIR, MODEL_PTH

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cache: dict = {}


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def build_efficientnet_b0():
    model = models.efficientnet_b0(weights=None)
    first = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, first.out_channels,
        kernel_size=first.kernel_size, stride=first.stride,
        padding=first.padding, bias=False
    )
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


MODEL_BUILDERS = {
    "Custom CNN":      CustomCNN,
    "ResNet18":        build_resnet18,
    "EfficientNet-B0": build_efficientnet_b0,
}


def load_model(model_name: str):
    if model_name in _cache:
        return _cache[model_name]

    builder = MODEL_BUILDERS.get(model_name)
    if builder is None:
        raise ValueError(f"Bilinmeyen model: {model_name}")

    weights_path = MODELS_DIR / MODEL_PTH[model_name]
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {weights_path}. "
            "Lütfen önce modeli eğitin ve .pth dosyasını backend/models/ klasörüne kopyalayın."
        )

    model = builder()
    state = torch.load(str(weights_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE).eval()

    _cache[model_name] = model
    return model


def preload_all_models():
    for name in MODEL_BUILDERS:
        try:
            load_model(name)
        except FileNotFoundError:
            pass
