from pathlib import Path

IMG_SIZE = 224
MEAN = 0.5099
STD = 0.2546
LABELS = ["Normal", "Zatürre (Pneumonia)"]

MODELS_DIR = Path(__file__).parent.parent.parent / "models"

MODEL_PTH = {
    "Custom CNN":      "Custom CNN.pth",
    "ResNet18":        "ResNet18.pth",
    "EfficientNet-B0": "EfficientNet-B0.pth",
}

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_MB = 10
