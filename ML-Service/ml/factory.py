from ultralytics import YOLO
from ml.models import EfficientNetWithEmbeddings
from ml.ensemble import FineGrainedEnsemble
from utils.config import Config


def create_ensemble_model() -> FineGrainedEnsemble:
    """
    Creates and initializes the FineGrainedEnsemble model using configuration from Config.
    """
    yolo = YOLO(Config.YOLO_MODEL)
    embedder = EfficientNetWithEmbeddings.load(Config.EMBEDDER_MODEL, num_classes=Config.EMBEDDER_NUM_CLASSES)
    screwdriver_model = EfficientNetWithEmbeddings.load(Config.SCREWDRIVER_MODEL, num_classes=Config.SCREWDRIVER_NUM_CLASSES)
    
    return FineGrainedEnsemble(
        detector=yolo,
        classifier=screwdriver_model,
        embedder=embedder,
        state_of_true=Config.SCREWDRIVER_STATE_OF_TRUE,
        alpha=Config.ALPHA,
        yolo_size=Config.YOLO_IMG_SIZE,
        device=Config.DEVICE
    )
