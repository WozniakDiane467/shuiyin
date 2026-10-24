from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

from .detector import WatermarkDetector
from .inpainter import LamaInpainter
from ..utils.image_utils import load_image, save_image, dilate_mask


class ImageProcessor:
    def __init__(self, device: str = None, cache_dir: str = None):
        self.device = device
        self.detector = WatermarkDetector(device=device, cache_dir=cache_dir)
        self.inpainter = LamaInpainter(device=device, cache_dir=cache_dir)

    def detect(self, image: Image.Image, prompt: str = "watermark, logo, text",
               max_bbox_percent: float = None) -> list:
        return self.detector.detect(image, prompt=prompt, max_bbox_percent=max_bbox_percent)

    def create_mask(self, image: Image.Image, regions: list, dilation: int = 5) -> Image.Image:
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        for region in regions:
            x1, y1, x2, y2 = region["bbox"]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], fill=255)
        if dilation > 0:
            mask_arr = np.array(mask)
            mask_arr = dilate_mask(mask_arr, dilation)
            mask = Image.fromarray(mask_arr)
        return mask

    def process(self, image: Image.Image, regions: list, dilation: int = 5) -> tuple[Image.Image, Image.Image]:
        mask = self.create_mask(image, regions, dilation)
        result = self.inpainter.inpaint(image, mask)
        return result, mask
