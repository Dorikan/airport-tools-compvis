from pathlib import Path
import torch
import os


class Config:
    # пути к моделям
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "weights"

    YOLO_MODEL = MODELS_DIR / os.getenv('YOLO_MODEL', 'yolo.pt')
    EMBEDDER_MODEL = MODELS_DIR / os.getenv('EMBEDDER_MODEL', 'embedder.pth')
    SCREWDRIVER_MODEL = MODELS_DIR / os.getenv('SCREWDRIVER_MODEL', 'screwdriver-b0.pth')

    # device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # настройки изображений
    YOLO_IMG_SIZE = os.getenv("YOLO_IMG_SIZE", 640)

    # ансамбль
    ALPHA = os.getenv('ALPHA', 0)
    SCREWDRIVER_STATE_OF_TRUE = os.getenv('STATE_OF_TRUE', 'true') == 'true'

    # сервис
    API_HOST = os.getenv("ML_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("ML_PORT", 8000))

    # классы объектов
    CLASSES_DICT = {
        0: 'adjustable pliers',
        1: 'adjustable wrench',
        2: 'brace',
        3: 'cross-head screwdriver',
        4: 'flathead screwdriver',
        5: 'offset ph screwdriver',
        6: 'oil opener',
        7: 'opened wrench',
        8: 'pliers',
        9: 'side cutters',
        10: 'wire twisting pliers'
    }
