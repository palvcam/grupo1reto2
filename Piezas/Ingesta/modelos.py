import cv2
import numpy as np
import time
import random

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def encode_jpg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("No se pudo codificar a JPG.")
    return buf.tobytes()

def decode_img(payload: bytes) -> np.ndarray:
    arr = np.frombuffer(payload, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("No se pudo decodificar la imagen.")
    return img

def simple_augment(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Augment ligero para generar variantes en streaming:
    - flip horizontal
    - brillo/contraste suave
    - blur suave ocasional
    """
    aug_tags = []

    out = img.copy()
    if random.random() < 0.5:
        out = cv2.flip(out, 1)
        aug_tags.append("flip")

    # brillo/contraste
    if random.random() < 0.7:
        alpha = random.uniform(0.9, 1.1)  # contraste
        beta = random.uniform(-10, 10)    # brillo
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)
        aug_tags.append(f"bc({alpha:.2f},{beta:.1f})")

    if random.random() < 0.2:
        out = cv2.GaussianBlur(out, (3, 3), 0)
        aug_tags.append("blur")

    tag = "+".join(aug_tags) if aug_tags else "none"
    return out, tag

def now_ms() -> int:
    return int(time.time() * 1000)
