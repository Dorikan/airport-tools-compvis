from PIL import Image
import io
import requests


async def load_image(url) -> Image.Image:
    if url.startswith("http"):
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:  # TEST PART
        img = Image.open(url).convert("RGB")

    return img
