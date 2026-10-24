import os
import torch
import numpy as np
from PIL import Image
from simple_lama_inpainting.utils.util import prepare_img_and_mask, download_model

from ..config import MODELS_DIR

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",
)


class LamaInpainter:
    def __init__(self, device: str = None, cache_dir: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = str(cache_dir or MODELS_DIR)
        self.model = None

    def load(self):
        if self.model is not None:
            return
        # Pre-download model if not present, using torch hub cache by default
        model_path = download_model(LAMA_MODEL_URL)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self.load()
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)

        image_tensor, mask_tensor = prepare_img_and_mask(image, mask, self.device)

        with torch.inference_mode():
            inpainted = self.model(image_tensor, mask_tensor)
            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)
            result = Image.fromarray(cur_res)

        # Crop back to original size (prepare_img_and_mask pads to modulo 8)
        original_size = image.size
        result = result.crop((0, 0, original_size[0], original_size[1]))
        return result
