import uvicorn
from fastapi import FastAPI

from api.v1 import endpoints
from utils.config import Config

app = FastAPI(title="ML Service", version="1.0")
app.include_router(endpoints.router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)