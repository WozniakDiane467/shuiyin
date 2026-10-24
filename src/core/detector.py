from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

from ..config import FLORENCE_MODEL_NAME, FLORENCE_REVISION, MODELS_DIR, DEFAULT_MAX_BBOX_PERCENT
from ..utils.image_utils import bbox_area_percent


class WatermarkDetector:
    def __init__(self, device: str = None, cache_dir: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = str(cache_dir or MODELS_DIR)
        self.processor = None
        self.model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        self.processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_NAME,
            revision=FLORENCE_REVISION,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_NAME,
            revision=FLORENCE_REVISION,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        ).to(self.device)
        self._loaded = True

    def detect(self, image: Image.Image, prompt: str = "watermark, logo, text",
               max_bbox_percent: float = None) -> list:
        self.load()
        max_bbox_percent = max_bbox_percent or DEFAULT_MAX_BBOX_PERCENT

        prompts = [p.strip() for p in prompt.split(",") if p.strip()]
        if not prompts:
            prompts = ["watermark"]

        all_regions = []

        # 1. Open-vocabulary detection for each prompt
        for p in prompts:
            regions = self._detect_open_vocab(image, p)
            all_regions.extend(regions)

        # 2. OCR region detection for text-like prompts
        if any(p in {"text", "watermark"} for p in prompts):
            all_regions.extend(self._detect_ocr_regions(image))

        # Filter and deduplicate
        filtered = []
        for region in all_regions:
            bbox = region["bbox"]
            if bbox_area_percent(bbox, (image.width, image.height)) > max_bbox_percent:
                continue
            filtered.append(region)

        return self._deduplicate(filtered)

    def _detect_open_vocab(self, image: Image.Image, prompt: str) -> list:
        task_prompt = "<OPEN_VOCABULARY_DETECTION>"
        text_input = task_prompt + prompt
        inputs = self.processor(text=text_input, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        results = parsed.get("<OPEN_VOCABULARY_DETECTION>", {})
        regions = []

        # Standard bboxes
        for bbox, label in zip(results.get("bboxes", []), results.get("bboxes_labels", [])):
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
            elif len(bbox) == 8:
                xs = bbox[0::2]
                ys = bbox[1::2]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            else:
                continue
            regions.append({"bbox": [int(x1), int(y1), int(x2), int(y2)], "label": label})

        # Polygons
        for polygon, label in zip(results.get("polygons", []), results.get("polygons_labels", [])):
            if not polygon or len(polygon[0]) < 2:
                continue
            pts = polygon[0]
            xs = pts[0::2]
            ys = pts[1::2]
            regions.append({
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "label": label,
            })

        return regions

    def _detect_ocr_regions(self, image: Image.Image) -> list:
        task_prompt = "<OCR_WITH_REGION>"
        inputs = self.processor(text=task_prompt, images=image, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        results = parsed.get("<OCR_WITH_REGION>", {})
        regions = []
        for quad, label in zip(results.get("quad_boxes", []), results.get("labels", [])):
            xs = quad[0::2]
            ys = quad[1::2]
            regions.append({
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "label": label.strip().replace("</s>", ""),
            })
        return regions

    def _deduplicate(self, regions: list, iou_threshold: float = 0.5) -> list:
        if not regions:
            return []
        regions = sorted(regions, key=lambda r: (r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1]), reverse=True)
        kept = []
        for r in regions:
            if all(self._iou(r["bbox"], k["bbox"]) < iou_threshold for k in kept):
                kept.append(r)
        return kept

    def _iou(self, box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0
