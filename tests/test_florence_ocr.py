import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

from src.config import FLORENCE_MODEL_NAME, MODELS_DIR


def create_test_image(path: Path):
    img = Image.new("RGB", (800, 600), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 100, 400, 400], fill=(255, 200, 150))
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.text((250, 250), "WATERMARK", fill=(255, 255, 255, 180), font=font)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.save(path, quality=95)
    return img


def test_task(image, task_prompt, text_input=None):
    processor = AutoProcessor.from_pretrained(FLORENCE_MODEL_NAME, cache_dir=str(MODELS_DIR), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(FLORENCE_MODEL_NAME, cache_dir=str(MODELS_DIR), trust_remote_code=True).to("cpu")

    if text_input:
        inputs = processor(text=text_input, images=image, return_tensors="pt").to("cpu")
    else:
        inputs = processor(text=task_prompt, images=image, return_tensors="pt").to("cpu")

    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))
    return parsed


def main():
    test_dir = project_root / "tests" / "fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    input_path = test_dir / "watermarked.jpg"
    image = create_test_image(input_path)

    print("Testing OCR...")
    print(test_task(image, "<OCR>"))

    print("\nTesting OCR with region...")
    print(test_task(image, "<OCR_WITH_REGION>"))

    print("\nTesting open vocab detection for watermark...")
    print(test_task(image, "<OPEN_VOCABULARY_DETECTION>", "<OPEN_VOCABULARY_DETECTION>watermark"))

    print("\nTesting open vocab detection for text...")
    print(test_task(image, "<OPEN_VOCABULARY_DETECTION>", "<OPEN_VOCABULARY_DETECTION>text"))


if __name__ == "__main__":
    main()
