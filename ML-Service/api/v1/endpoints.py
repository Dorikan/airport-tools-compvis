from fastapi import APIRouter, Request, Query
from services.prediction_service import PredictionService
from services.batched_prediction_service import BatchedPredictionService

router = APIRouter()
service = BatchedPredictionService()


@router.get("/predict")
async def predict(
        request: Request,
        url: str = Query(..., description="URL изображения для анализа"),
        image_id: str = Query(..., description="ID входного изображения"),
        threshold: float = Query(0.8, description="Threshold для эмбеддинга")
):
    result = await service.predict(url, threshold=threshold)
    result["image_id"] = image_id

    return result
