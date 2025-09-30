from PIL import Image
from ultralytics import YOLO

from ml.preprocessing import load_image
from ml.ensemble import FineGrainedEnsemble
from utils.config import Config
from ml.models import EfficientNetWithEmbeddings
from utils.draw import draw_results
from io import BytesIO
import base64


class PredictionService:
    def __init__(self):

        self.yolo = YOLO(Config.YOLO_MODEL)
        self.embedder = EfficientNetWithEmbeddings.load(Config.EMBEDDER_MODEL)
        self.screwdriver_model = EfficientNetWithEmbeddings.load(Config.SCREWDRIVER_MODEL, num_classes=3)
        self.model = FineGrainedEnsemble(
            self.yolo, self.screwdriver_model, self.embedder,
            state_of_true=True, alpha=Config.ALPHA, yolo_size=Config.YOLO_IMG_SIZE,
            device=Config.DEVICE
        )

    async def predict(self, url, threshold=0.8):
        image = await load_image(url)
        result = self.model.predict(image, threshold=threshold)

        debug_image = Image.fromarray(await draw_results(image, result))
        buffer = BytesIO()
        debug_image.save(buffer, format='JPEG')
        base64_image = base64.b64encode(buffer.getvalue())
        debug_image.save('temp.jpg', format='JPEG')

        return {'instruments': result, 'debug_image': base64_image}
