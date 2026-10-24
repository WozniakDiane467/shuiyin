from pathlib import Path
from PIL import Image
import numpy as np
import cv2


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: str | Path, quality: int = 95):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        image.save(path, quality=quality)
    else:
        image.save(path)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def dilate_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(mask, kernel, iterations=1)


def bbox_area_percent(bbox, image_size):
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    return area / (image_size[0] * image_size[1]) * 100
