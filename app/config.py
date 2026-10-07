import os

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_head_only.pth")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "5"))
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/bmp"}
MIN_IMAGE_SIDE = 32          # reject tiny/garbage images
MAX_IMAGE_SIDE = 4096        # reject huge images (memory protection)