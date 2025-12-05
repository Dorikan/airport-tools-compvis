import asyncio
from PIL import Image
from ml.preprocessing import load_image
from utils.config import Config
from utils.draw import draw_results
from ml.factory import create_ensemble_model
from fastapi import HTTPException
from io import BytesIO
import base64


class PredictionService:
    """
    Service for single image prediction using an ensemble of models.
    """
    def __init__(self):
        """
        Initializes the prediction service by creating the ensemble model.
        """
        self.model = create_ensemble_model()

    async def predict(self, url: str, threshold: float = Config.THRESHOLD) -> dict:
        """
        Predicts instruments in the image from the given URL.

        Args:
            url (str): URL of the image to analyze.
            threshold (float): Confidence threshold for embedding generation.

        Returns:
            dict: A dictionary containing the prediction results and a base64 encoded debug image.
        """
        try:
            image = await load_image(url)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.model.predict, image, threshold)

            debug_image = Image.fromarray(await draw_results(image, result))
            (width, height) = (debug_image.width // 2, debug_image.height // 2)
            debug_image = debug_image.resize((width, height))
            buffer = BytesIO()
            debug_image.save(buffer, format='JPEG')
            base64_image = base64.b64encode(buffer.getvalue())

            return {'instruments': result, 'debug_image': base64_image}
        except Exception as e:
            print(f"Error processing image {url}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process image: {str(e)}")
