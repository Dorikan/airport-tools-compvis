import os
import cv2

images_dir = "dataset/images/train"
labels_dir = "dataset/labels/train"
output_dir = "dataset_crops/train"

os.makedirs(output_dir, exist_ok=True)


def yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    x1 = int((x_center - w / 2) * img_w)
    y1 = int((y_center - h / 2) * img_h)
    x2 = int((x_center + w / 2) * img_w)
    y2 = int((y_center + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)


for img_file in os.listdir(images_dir):
    if not img_file.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(images_dir, img_file)
    label_path = os.path.join(labels_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        cls, xc, yc, bw, bh = map(float, line.strip().split())
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        crop = img[y1:y2, x1:x2]

        class_dir = os.path.join(output_dir, str(int(cls)))
        os.makedirs(class_dir, exist_ok=True)
        out_path = os.path.join(class_dir, f"{os.path.splitext(img_file)[0]}_{i}.jpg")
        cv2.imwrite(out_path, crop)
