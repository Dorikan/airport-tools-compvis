import asyncio
import base64
from io import BytesIO
from collections import deque
from typing import List

from PIL import Image
from ultralytics.models import YOLO

from ml.ensemble import FineGrainedEnsemble
from ml.models import EfficientNetWithEmbeddings
from ml.preprocessing import load_image
from utils.config import Config
from utils.draw import draw_results


class BatchedPredictionService:
    def __init__(self, max_batch_size=8, batch_timeout=0.05):
        self.yolo = YOLO(Config.YOLO_MODEL)
        self.embedder = EfficientNetWithEmbeddings.load(Config.EMBEDDER_MODEL)
        self.screwdriver_model = EfficientNetWithEmbeddings.load(Config.SCREWDRIVER_MODEL, num_classes=3)
        self.model = FineGrainedEnsemble(
            self.yolo, self.screwdriver_model, self.embedder,
            state_of_true=True, alpha=Config.ALPHA, yolo_size=Config.YOLO_IMG_SIZE,
            device=Config.DEVICE
        )
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.queue = deque()
        self.lock = asyncio.Lock()
        asyncio.create_task(self._batch_worker())

    async def predict(self, url, threshold=0.8):
        loop = asyncio.get_event_loop()
        fut = loop.create_future()

        async with self.lock:
            self.queue.append((url, threshold, fut))

        return await fut

    async def _batch_worker(self):
        while True:
            await asyncio.sleep(self.batch_timeout)

            async with self.lock:
                if not self.queue:
                    continue
                batch = [self.queue.popleft() for _ in range(min(self.max_batch_size, len(self.queue)))]

            urls, thresholds, futures = zip(*batch)

            images: List[Image.Image] = await asyncio.gather(*[load_image(url) for url in urls])

            results = self.model.predict_batch(images, thresholds)

            for fut, image, result in zip(futures, images, results):
                debug_image = Image.fromarray(await draw_results(image, result))

                (width, height) = (debug_image.width // 2, debug_image.height // 2)
                debug_image = debug_image.resize((width, height))

                buffer = BytesIO()
                debug_image.save(buffer, format='JPEG')
                debug_image.save("result.jpg", format='JPEG')
                base64_image = base64.b64encode(buffer.getvalue())

                fut.set_result({
                    'instruments': result,
                    'debug_image': base64_image
                })