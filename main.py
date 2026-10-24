import os
import sys
from pathlib import Path

# PyInstaller onefile: C++ 扩展（如 torchvision/_C_stable.pyd）依赖 torch/lib 下的 DLL，
# 但单文件 exe 解压后这些 DLL 在子目录中，Windows 默认搜不到。提前把 torch/lib 加入搜索路径。
if getattr(sys, 'frozen', False):
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        torch_lib = Path(meipass) / 'torch' / 'lib'
        if torch_lib.is_dir():
            try:
                os.add_dll_directory(str(torch_lib))
            except (AttributeError, OSError):
                pass

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.app import run

if __name__ == "__main__":
    run()
