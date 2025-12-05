import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from services.batched_prediction_service import BatchedPredictionService
import warnings

from api.v1 import endpoints
from utils.config import Config

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.batched_service = BatchedPredictionService()
    yield

app = FastAPI(title="ML Service", version="1.0", lifespan=lifespan)
app.include_router(endpoints.router, prefix="/api/v1")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="Could not initialize NNPACK")
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)