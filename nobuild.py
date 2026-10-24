import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN = PROJECT_ROOT / "main.py"
DIST = PROJECT_ROOT / "dist"
BUILD = PROJECT_ROOT / "build"
ICON = PROJECT_ROOT / "assets" / "icon.ico"


def clean():
    for d in [BUILD, PROJECT_ROOT / "qu-shuiyin.spec"]:
        if d.exists():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
            else:
                try:
                    d.unlink()
                except OSError:
                    pass

    # dist 目录保留，只删除本次要生成的同名 exe，避免 bak/历史文件被锁导致打包失败
    target_exe = DIST / "qu-shuiyin.exe"
    if target_exe.exists():
        try:
            target_exe.unlink()
        except OSError as e:
            print(f"Warning: could not remove old exe {target_exe}: {e}")


def build():
    clean()
    DIST.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "qu-shuiyin",
        "--onefile",
        "--windowed",
        "--distpath", str(DIST),
        "--workpath", str(BUILD),
        # 关键修复：torchvision/torch/transformers 有大量动态库和元数据，必须完整收集
        "--collect-all", "torch",
        "--collect-all", "torchvision",
        "--collect-all", "transformers",
        "--collect-all", "simple_lama_inpainting",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "torchvision.io",
        "--hidden-import", "transformers",
        "--hidden-import", "transformers.models.auto.processing_auto",
        "--copy-metadata", "torch",
        "--copy-metadata", "torchvision",
        "--copy-metadata", "transformers",
    ]

    if ICON.exists():
        cmd.extend(["--icon", str(ICON)])

    cmd.append(str(MAIN))

    subprocess.run(cmd, check=True)
    print(f"Build complete: {DIST / 'qu-shuiyin.exe'}")


if __name__ == "__main__":
    build()
