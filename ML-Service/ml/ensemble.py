import numpy as np
import torch
from ultralytics import YOLO
from ml.models import EfficientNetWithEmbeddings
from ml.transforms.efficientnet import EfficientNetTransforms


class FineGrainedEnsemble:
    def __init__(self, detector: YOLO, classifier: EfficientNetWithEmbeddings, embedder: EfficientNetWithEmbeddings,
                 state_of_true=False, alpha=0.7, yolo_size=640, device="cpu"):
        self.detector = detector.to(device)
        self.classifier = classifier.to(device)
        self.embedder = embedder.to(device)
        self.device = device
        self.state_of_true = state_of_true
        self.alpha = alpha
        self.yolo_size = yolo_size
        self.efficientnet_tfs = EfficientNetTransforms()

    def __refine_screwdriver(self, x):
        with torch.no_grad():
            logits = self.classifier(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probs))

            return pred + 3, probs[pred]

    def predict(self, image, threshold=0.8):
        r = self.detector(image, imgsz=self.yolo_size, agnostic_nms=True, retina_masks=True)[0]

        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        result = []

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = self.efficientnet_tfs(crop).unsqueeze(0).to(self.device)

            if cls in [3, 4, 5]:
                if self.state_of_true:
                    cls, conf = self.__refine_screwdriver(crop_tensor)
                else:
                    yolo_probs = np.zeros(len(r.names))
                    yolo_probs[int(cls)] = conf
                    pred_, conf_ = self.__refine_screwdriver(crop_tensor)
                    t = np.zeros(len(r.names))
                    t[pred_] = conf_

                    final_probs = self.alpha * yolo_probs + (1 - self.alpha) * t
                    cls = np.argmax(final_probs)
                    conf = final_probs[cls]

            embedding = None
            if conf < threshold:
                _, embedding = self.embedder(crop_tensor)
                embedding = embedding.cpu().squeeze(0).tolist()

            result.append({
                'bbox': box.tolist(),
                'class': int(cls),
                'confidence': float(conf),
                'embedding': embedding if embedding is not None else []
            })

        return result
