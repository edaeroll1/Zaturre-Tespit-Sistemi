import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path

# ── Sabitler (notebook ile aynı) ────────────────────────────────────────────
IMG_SIZE   = 224
MEAN       = 0.5099
STD        = 0.2546
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS     = ["Normal", "Zatürre (Pneumonia)"]
SCRIPT_DIR = Path(__file__).parent

# ── Model tanımları ──────────────────────────────────────────────────────────
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
    model.fc    = nn.Linear(model.fc.in_features, 2)
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

MODEL_PTH = {
    "Custom CNN":      "Custom CNN.pth",
    "ResNet18":        "ResNet18.pth",
    "EfficientNet-B0": "EfficientNet-B0.pth",
}

# ── Transform ────────────────────────────────────────────────────────────────
eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[MEAN], std=[STD]),
])

# ── Model yükleme (önbellek) ─────────────────────────────────────────────────
_cache: dict = {}


def load_model(model_name: str, weights_path: str):
    key = (model_name, weights_path)
    if key in _cache:
        return _cache[key], None

    builder = MODEL_BUILDERS.get(model_name)
    if builder is None:
        return None, f"Bilinmeyen model: {model_name}"

    model = builder()
    try:
        state = torch.load(weights_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state)
    except Exception as e:
        return None, f"Ağırlık dosyası yüklenemedi:\n{e}"

    model.to(DEVICE).eval()
    _cache[key] = model
    return model, None


# ── Tahmin fonksiyonu ────────────────────────────────────────────────────────
def predict(image, model_name: str):
    if image is None:
        return None, "Lütfen bir röntgen görüntüsü yükleyin."

    weights_path = SCRIPT_DIR / MODEL_PTH[model_name]
    if not weights_path.exists():
        return None, f"Model dosyası bulunamadı:\n{weights_path}\n\nLütfen önce notebook'u çalıştırarak modeli eğitin."

    model, err = load_model(model_name, str(weights_path))
    if err:
        return None, err

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    tensor = eval_transform(image.convert("L")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

    label = LABELS[int(probs.argmax())]
    confidence = float(probs.max()) * 100

    result_text = (
        f"**Tahmin: {label}**\n\n"
        f"Normal olasılığı: {probs[0]*100:.1f}%\n"
        f"Zatürre olasılığı: {probs[1]*100:.1f}%\n\n"
        f"Güven skoru: {confidence:.1f}%"
    )

    return {LABELS[0]: float(probs[0]), LABELS[1]: float(probs[1])}, result_text


# ── Gradio arayüzü ───────────────────────────────────────────────────────────
with gr.Blocks(title="Zatürre Tespit Sistemi") as demo:
    gr.Markdown(
        """
        # Zatürre Tespit Sistemi
        Göğüs röntgeni (chest X-ray) görüntüsünü yükleyerek **Normal** ya da **Zatürre (Pneumonia)**
        tespiti yapabilirsiniz.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Röntgen Görüntüsü",
                type="pil",
                height=300,
            )
            model_selector = gr.Dropdown(
                choices=list(MODEL_BUILDERS.keys()),
                value="EfficientNet-B0",
                label="Model Seçin",
            )
            predict_btn = gr.Button("Tahmin Et", variant="primary")

        with gr.Column(scale=1):
            label_output  = gr.Label(label="Tahmin Sonuçları", num_top_classes=2)
            detail_output = gr.Markdown(label="Detay")

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_selector],
        outputs=[label_output, detail_output],
    )

    gr.Markdown(
        """
        ---
        **Model Performansları (test seti):**

        | Model | Accuracy | F1-Score | Recall |
        |---|---|---|---|
        | Custom CNN | 88.5% | 88.9% | 92.2% |
        | ResNet18 | 91.6% | 92.1% | 98.8% |
        | EfficientNet-B0 | 91.6% | **92.2%** | **99.1%** |
        """
    )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), inbrowser=False)
