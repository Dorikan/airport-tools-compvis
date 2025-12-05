from fastapi import APIRouter, Request, Query
from services.prediction_service import PredictionService
from services.batched_prediction_service import BatchedPredictionService

router = APIRouter()



@router.get("/predict")
async def predict(
        request: Request,
        url: str = Query(..., description="URL изображения для анализа"),
        image_id: str = Query(..., description="ID входного изображения"),
        threshold: float = Query(0.8, description="Threshold для эмбеддинга")
):
    """
    Endpoint to predict instruments in an image.

    Args:
        request (Request): The FastAPI request object.
        url (str): URL of the image to analyze.
        image_id (str): ID of the input image.
        threshold (float): Confidence threshold for embedding generation.

    Returns:
        dict: Prediction results including detected instruments and a debug image.
    """
    service: BatchedPredictionService = request.app.state.batched_service

    result = await service.predict(url, threshold=threshold)
    result["image_id"] = image_id

    return result
