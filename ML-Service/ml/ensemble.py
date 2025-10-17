import numpy as np
import torch
from ultralytics import YOLO
from ml.models import EfficientNetWithEmbeddings
from ml.transforms.efficientnet import EfficientNetTransforms


class FineGrainedEnsemble:
    def __init__(self, detector: YOLO, classifier: EfficientNetWithEmbeddings, embedder: EfficientNetWithEmbeddings,
                 state_of_true=False, alpha=0.7, yolo_size=(640, 640), device="cpu"):
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
        r = self.detector(image, imgsz=self.yolo_size, agnostic_nms=True, retina_masks=True, half=True)[0]

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
                _, embedding = self.embedder(crop_tensor, return_embedding=True)
                embedding = embedding.cpu().squeeze(0).tolist()

            result.append({
                'bbox': box.tolist(),
                'class': int(cls),
                'confidence': float(conf),
                'embedding': embedding if embedding is not None else []
            })

        return result

    def predict_batch(self, images, thresholds):
        results = self.detector(images, imgsz=self.yolo_size, agnostic_nms=True, retina_masks=True)
        all_results = []

        for image, result, threshold in zip(images, results, thresholds):
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            if len(boxes) == 0:
                all_results.append([])
                continue

            crops = [image.crop((x1, y1, x2, y2)) for (x1, y1, x2, y2) in boxes]
            x = [self.efficientnet_tfs(crop) for crop in crops]
            batch_tensors = torch.stack(x).to(self.device)

            screw_mask = [cls in [3, 4, 5] for cls in classes]
            if any(screw_mask):
                with torch.no_grad():
                    screw_tensors = batch_tensors[screw_mask]
                    logits_screw = self.classifier(screw_tensors)
                    screw_probs = torch.softmax(logits_screw, dim=1).cpu().numpy()
            else:
                screw_probs = np.zeros((0, 3))

            low_conf_mask = [conf < threshold for conf in confs]
            if any(low_conf_mask):
                low_conf_tensors = batch_tensors[low_conf_mask]
                with torch.no_grad():
                    _, emb_low = self.embedder(low_conf_tensors, return_embedding=True)
                emb_low = emb_low.cpu().numpy()
            else:
                emb_low = np.zeros((0, 1024))

            # === Формируем финальный результат
            detections = []
            screw_ptr, emb_ptr = 0, 0
            for box, conf, cls, screw_flag, low_flag in zip(boxes, confs, classes, screw_mask, low_conf_mask):
                if screw_flag:
                    if self.state_of_true:
                        pred = int(np.argmax(screw_probs[screw_ptr]))
                        conf = float(screw_probs[screw_ptr, pred])
                        cls = pred + 3
                    else:
                        yolo_probs = np.zeros(11)
                        yolo_probs[cls] = conf
                        refine = screw_probs[screw_ptr]
                        refine_probs = np.zeros(11)
                        refine_probs[3:6] = refine
                        final_probs = self.alpha * yolo_probs + (1 - self.alpha) * refine_probs
                        cls = int(np.argmax(final_probs))
                        conf = float(final_probs[cls])
                    screw_ptr += 1

                embedding = emb_low[emb_ptr].tolist() if low_flag else []
                if low_flag:
                    emb_ptr += 1

                detections.append({
                    "bbox": box.tolist(),
                    "class": int(cls),
                    "confidence": float(conf),
                    "embedding": embedding
                })

            all_results.append(detections)

        return all_results
