from PIL import Image
import io
import aiohttp


async def load_image(url: str) -> Image.Image:
    """
    Asynchronously loads an image from a URL or local path.

    Args:
        url (str): URL or local path to the image.

    Returns:
        Image.Image: Loaded PIL Image converted to RGB.
    """
    if url.startswith("http"):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.read()
                img = Image.open(io.BytesIO(data)).convert("RGB")
    else:  # TEST PART
        img = Image.open(url).convert("RGB")

    return img
