import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def _make_image_bytes(mode="RGB", size=(224, 224)) -> bytes:
    img = Image.new(mode, size, color=128)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root():
    response = client.get("/")
    assert response.status_code == 200


@pytest.mark.parametrize("model_name", ["Custom CNN", "ResNet18", "EfficientNet-B0"])
def test_predict_valid(model_name):
    image_bytes = _make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        data={"model_name": model_name},
    )
    assert response.status_code == 200
    body = response.json()
    assert "label" in body
    assert "confidence" in body
    assert "normal_prob" in body
    assert "pneumonia_prob" in body
    assert body["model_name"] == model_name
    assert 0 <= body["confidence"] <= 100


def test_predict_invalid_model():
    image_bytes = _make_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        data={"model_name": "YanlisModel"},
    )
    assert response.status_code == 400


def test_predict_invalid_format():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"bu bir metin dosyasidir", "text/plain")},
        data={"model_name": "EfficientNet-B0"},
    )
    assert response.status_code == 415


def test_predict_corrupted_image():
    response = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"bozuk veri", "image/jpeg")},
        data={"model_name": "EfficientNet-B0"},
    )
    assert response.status_code == 400
