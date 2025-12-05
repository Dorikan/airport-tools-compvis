import asyncio
import base64
from io import BytesIO
from collections import deque
from typing import List

from PIL import Image

from ml.preprocessing import load_image
from utils.config import Config
from utils.draw import draw_results
from ml.factory import create_ensemble_model


class BatchedPredictionService:
    """
    Service for batched image prediction.
    Accumulates requests and processes them in batches to optimize model inference.
    """
    def __init__(self, max_batch_size=Config.MAX_BATCH_SIZE, batch_timeout=Config.BATCH_TIMEOUT):
        """
        Initializes the batched prediction service.

        Args:
            max_batch_size (int): Maximum number of images in a single batch.
            batch_timeout (float): Maximum time to wait before processing a partial batch.
        """
        self.model = create_ensemble_model()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue = deque()
        self.lock = asyncio.Lock()
        asyncio.create_task(self._batch_worker())

    async def predict(self, url, threshold=Config.THRESHOLD):
        """
        Queues a prediction request and awaits the result.

        Args:
            url (str): URL of the image to analyze.
            threshold (float): Confidence threshold for embedding generation.

        Returns:
            dict: A dictionary containing the prediction results and a base64 encoded debug image.
        """
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        async with self.lock:
            self.queue.append((url, threshold, fut))

        return await fut

    async def _batch_worker(self):
        """
        Background task that continuously processes the queue of prediction requests.
        """
        while True:
            await asyncio.sleep(self.batch_timeout)

            async with self.lock:
                if not self.queue:
                    continue
                batch = [self.queue.popleft() for _ in range(min(self.max_batch_size, len(self.queue)))]

            urls, thresholds, futures = zip(*batch)

            try:
                images: List[Image.Image] = await asyncio.gather(*[load_image(url) for url in urls])

                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(None, self.model.predict_batch, images, thresholds)

                for fut, image, result in zip(futures, images, results):
                    if fut.done(): continue
                    try:
                        debug_image = Image.fromarray(await draw_results(image, result))

                        (width, height) = (debug_image.width // 2, debug_image.height // 2)
                        debug_image = debug_image.resize((width, height))

                        buffer = BytesIO()
                        debug_image.save(buffer, format='JPEG')
                        base64_image = base64.b64encode(buffer.getvalue())

                        fut.set_result({
                            'instruments': result,
                            'debug_image': base64_image
                        })
                    except Exception as e:
                        fut.set_exception(e)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)