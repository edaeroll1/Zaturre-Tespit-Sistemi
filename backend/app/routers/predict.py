from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image, UnidentifiedImageError
import io

from ..schemas.prediction import PredictionResponse
from ..services.predict_service import predict
from ..core.config import ALLOWED_CONTENT_TYPES, MAX_FILE_SIZE_MB, MODEL_PTH

router = APIRouter(prefix="/predict", tags=["Tahmin"])


@router.post("", response_model=PredictionResponse)
async def predict_endpoint(
    file: UploadFile = File(..., description="Göğüs röntgeni görüntüsü (JPG veya PNG)"),
    model_name: str = Form("EfficientNet-B0", description="Kullanılacak model"),
):
    if model_name not in MODEL_PTH:
        raise HTTPException(
            status_code=400,
            detail=f"Geçersiz model adı. Seçenekler: {list(MODEL_PTH.keys())}"
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Desteklenmeyen dosya formatı: {file.content_type}. Sadece JPG ve PNG kabul edilir."
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Dosya boyutu {MAX_FILE_SIZE_MB}MB sınırını aşıyor."
        )

    try:
        image = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Geçersiz görüntü dosyası.")

    try:
        result = predict(image, model_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PredictionResponse(**result)
