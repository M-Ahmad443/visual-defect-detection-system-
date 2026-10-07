import io
import logging
import math

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger("casting-api")

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class DefectClassifier:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.threshold = float(ckpt["threshold"])
        self.img_size = int(ckpt["img_size"])
        self.classes = ckpt["classes"]            # ["normal", "defective"]

        # Must match the training-time architecture exactly
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(model.classifier[1].in_features, 1),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self.model = model

        self.tf = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
        logger.info("Model loaded | threshold=%.3f img_size=%d",
                    self.threshold, self.img_size)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        x = self.tf(image.convert("RGB")).unsqueeze(0)
        logit = self.model(x).squeeze().item()
        p_defect = 1 / (1 + math.exp(-logit))     # raw model probability

        is_defect = p_defect >= self.threshold
        # Re-center the score so the tuned threshold maps to 0.5.
        # Otherwise p=0.10 with threshold 0.064 would say "defective, 10% confident".
        score = 1 / (1 + math.exp(-(logit - _logit(self.threshold))))
        confidence = score if is_defect else 1 - score

        return {
            "predicted_class": self.classes[1] if is_defect else self.classes[0],
            "confidence": round(confidence, 4),
            "defect_probability": round(p_defect, 4),
            "threshold": round(self.threshold, 4),
        }


def load_image(data: bytes, min_side: int, max_side: int) -> Image.Image:
    """Decode and validate image bytes. Raises ValueError with a clear message."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()                              # checks file integrity
        img = Image.open(io.BytesIO(data))        # re-open after verify()
        img.load()
    except Exception:
        raise ValueError("File is not a valid or readable image.")

    w, h = img.size
    if min(w, h) < min_side:
        raise ValueError(f"Image too small ({w}x{h}); minimum side is {min_side}px.")
    if max(w, h) > max_side:
        raise ValueError(f"Image too large ({w}x{h}); maximum side is {max_side}px.")
    return img