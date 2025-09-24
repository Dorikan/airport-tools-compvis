import io
from datetime import datetime

import torch
import uvicorn
import requests
import numpy as np
from PIL import Image
from fastapi import FastAPI, Query
from torchvision import transforms
from models import EfficientNetWithEmbeddings
from ultralytics import YOLO
import debug


app = FastAPI(
    title="Instrument QC Service",
    description="YOLO11 + EfficientNet hybrid pipeline with debug outputs",
    version="1.0"
)


yolo_model = YOLO("weights/yolo/best.pt")
efficientnet_model = EfficientNetWithEmbeddings(num_classes=11).load("weights/efficientnet/best_model.pth")

CONF_THRESHOLD = 0.9


eff_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

yolo_size = 640


@app.get("/analyze/")
async def analyze_image(
    url: str = Query(..., description="URL изображения для анализа")
):
    # Загружаем картинку
    if url.startswith("http"):
        response = requests.get(url)
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        img = Image.open(url).convert("RGB")

    # YOLO resize 640x640
    img_resized = img.resize((yolo_size, yolo_size))

    results = yolo_model(img_resized, imgsz=yolo_size)

    final_output = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls in zip(boxes, confs, classes):
            if conf < 0.75:
                continue
            x1, y1, x2, y2 = map(int, box)
            cropped = img_resized.crop((x1, y1, x2, y2))

            x = eff_transform(cropped).unsqueeze(0)
            with torch.no_grad():
                logits, embed = efficientnet_model(x, return_embedding=True)
                embed = embed.squeeze(0).tolist()
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                if conf < CONF_THRESHOLD:
                    cls = int(np.argmax(probs))
                    conf = float(np.max(probs))


            final_output.append({
                "bbox": [x1, y1, x2, y2],
                "class": int(cls),
                "confidence": float(conf),
                "embedding": embed
            })

    # ---- Сохраняем debug-картинку ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug.draw_boxes_and_save(img_resized, final_output, f"debug_{timestamp}.jpg")

    return {"results": final_output}


# ---- Запуск ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
