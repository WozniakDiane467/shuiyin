import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from PIL import Image, ImageDraw, ImageFont
from src.core.processor import ImageProcessor
from src.utils.image_utils import save_image


def create_test_image(path: Path):
    img = Image.new("RGB", (800, 600), color=(135, 206, 235))
    draw = ImageDraw.Draw(img)
    # Background pattern
    for i in range(0, 800, 40):
        draw.line([(i, 0), (i, 600)], fill=(120, 190, 220), width=1)
    for i in range(0, 600, 40):
        draw.line([(0, i), (800, i)], fill=(120, 190, 220), width=1)
    # Some shapes
    draw.rectangle([100, 100, 400, 400], fill=(255, 200, 150))
    draw.ellipse([450, 150, 650, 350], fill=(100, 150, 200))
    # Watermark text
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except:
        font = ImageFont.load_default()
    # Draw semi-transparent watermark
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.text((250, 250), "WATERMARK", fill=(255, 255, 255, 180), font=font)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    img.save(path, quality=95)
    return img


def main():
    test_dir = project_root / "tests" / "fixtures"
    test_dir.mkdir(parents=True, exist_ok=True)
    input_path = test_dir / "watermarked.jpg"
    output_path = test_dir / "clean.jpg"
    mask_path = test_dir / "mask.jpg"

    print("Creating test image...")
    image = create_test_image(input_path)

    print("Loading processor (first run downloads models)...")
    processor = ImageProcessor()

    print("Detecting watermark...")
    regions = processor.detect(image, prompt="watermark, logo, text", max_bbox_percent=20)
    print(f"Detected {len(regions)} regions: {regions}")

    if not regions:
        print("No regions detected, test aborted.")
        return

    print("Inpainting...")
    result, mask = processor.process(image, regions, dilation=5)

    save_image(result, output_path)
    save_image(mask, mask_path)
    print(f"Saved result to {output_path}")
    print(f"Saved mask to {mask_path}")


if __name__ == "__main__":
    main()
