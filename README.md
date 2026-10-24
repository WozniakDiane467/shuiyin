# qu-shuiyin

一个本地离线运行的图片去水印/去 Logo 桌面小工具。

## 核心能力

- 自动检测图片中的水印、文字、Logo 区域
- 支持手动补充或修正检测区域
- 单张预览处理效果
- 批量处理整个文件夹
- 批量模式先抽查前 3 张，确认效果后再跑完全部

## 技术栈

- PySide6：桌面界面
- Florence-2：水印/文字/Logo 自动检测
- LaMa：图像修复填充
- PyInstaller：打包 exe

## 运行方式

```bash
# 1. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动应用
python main.py
```

首次启动会自动下载 Florence-2 和 LaMa 模型到本地缓存，请保持网络畅通。
模型下载完成后即可离线使用。

## 打包 exe

```bash
python build.py
```

打包后的 exe 位于 `dist/qu-shuiyin.exe`。
首次运行 exe 时同样会自动下载模型。
