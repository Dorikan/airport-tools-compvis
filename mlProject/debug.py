from PIL import Image
import numpy as np
import cv2
import os


DEBUG_DIR = "debug_outputs"
os.makedirs(DEBUG_DIR, exist_ok=True)


def draw_boxes_and_save(image: Image.Image, detections, filename: str):
    """Сохраняет debug картинку с боксами"""
    img_cv = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR для OpenCV

    save_path = os.path.join(DEBUG_DIR, filename)

    cv2.imwrite(os.path.join(DEBUG_DIR, '/crops/', filename), img_cv)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cls = det["class"]
        conf = det["confidence"]

        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_cv,
            f"{cls}:{conf:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 0),
            2
        )

    cv2.imwrite(save_path, img_cv)

    return save_path
