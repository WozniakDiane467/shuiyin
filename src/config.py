import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    # Running inside PyInstaller bundle
    PROJECT_ROOT = Path(sys.executable).resolve().parent
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_MAX_BBOX_PERCENT = 15
DEFAULT_INPAINT_DILATION = 5

FLORENCE_MODEL_NAME = "microsoft/Florence-2-base-ft"
FLORENCE_REVISION = "main"

DETECTION_PROMPTS = ["watermark", "logo", "text"]
SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
