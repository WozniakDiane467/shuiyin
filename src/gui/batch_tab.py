from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
    QGroupBox, QSpinBox, QLineEdit, QSplitter
)
from PySide6.QtCore import Qt, QThread, Signal
from PIL import Image

from .image_view import ImageView
from ..core.processor import ImageProcessor
from ..utils.image_utils import load_image, save_image
from ..config import SUPPORTED_IMAGE_EXTS


class BatchProcessThread(QThread):
    progress = Signal(int, int, str)
    preview = Signal(object, str)
    finished = Signal(int, int)
    error = Signal(str)

    def __init__(self, processor, image_paths, input_dir, output_dir, prompt, max_bbox_percent, dilation,
                 start_index=0, preview_count=0, preview_stop_at=-1):
        super().__init__()
        self.processor = processor
        self.image_paths = image_paths
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.prompt = prompt
        self.max_bbox_percent = max_bbox_percent
        self.dilation = dilation
        self.start_index = start_index
        self.preview_count = preview_count
        self.preview_stop_at = preview_stop_at
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            total = len(self.image_paths)
            processed = 0
            previewed = 0
            for i in range(self.start_index, total):
                if self._stop:
                    break
                path = self.image_paths[i]
                rel = path.relative_to(self.input_dir)
                self.progress.emit(i + 1, total, str(rel))

                try:
                    image = load_image(path)
                    regions = self.processor.detect(image, self.prompt, self.max_bbox_percent)
                    if regions:
                        result, _ = self.processor.process(image, regions, self.dilation)
                    else:
                        result = image

                    out_path = self.output_dir / rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    save_image(result, out_path)

                    if previewed < self.preview_count:
                        self.preview.emit(result, str(rel))
                        previewed += 1
                        if previewed >= self.preview_count and self.preview_stop_at == i:
                            self.finished.emit(i + 1, total)
                            return

                    processed += 1
                except Exception as e:
                    self.error.emit(f"{path.name}: {e}")

            self.finished.emit(total, total)
        except Exception as e:
            self.error.emit(str(e))


class BatchTab(QWidget):
    def __init__(self, processor: ImageProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.input_dir = None
        self.output_dir = None
        self.batch_thread = None
        self.image_paths = []
        self.preview_queue = []
        self.preview_mode = False
        self.preview_stop_index = -1

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.btn_input = QPushButton("选择输入文件夹")
        self.btn_input.clicked.connect(self.select_input)
        left_layout.addWidget(self.btn_input)
        self.lbl_input = QLabel("未选择")
        self.lbl_input.setWordWrap(True)
        left_layout.addWidget(self.lbl_input)

        self.btn_output = QPushButton("选择输出文件夹")
        self.btn_output.clicked.connect(self.select_output)
        left_layout.addWidget(self.btn_output)
        self.lbl_output = QLabel("未选择")
        self.lbl_output.setWordWrap(True)
        left_layout.addWidget(self.lbl_output)

        settings = QGroupBox("检测设置")
        settings_layout = QVBoxLayout(settings)
        settings_layout.addWidget(QLabel("检测提示词"))
        self.prompt_edit = QLineEdit("watermark, logo, text")
        settings_layout.addWidget(self.prompt_edit)
        settings_layout.addWidget(QLabel("最大水印区域占比(%)"))
        self.max_bbox_spin = QSpinBox()
        self.max_bbox_spin.setRange(1, 50)
        self.max_bbox_spin.setValue(15)
        settings_layout.addWidget(self.max_bbox_spin)
        settings_layout.addWidget(QLabel("蒙版膨胀半径"))
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(0, 30)
        self.dilation_spin.setValue(5)
        settings_layout.addWidget(self.dilation_spin)
        left_layout.addWidget(settings)

        self.btn_start = QPushButton("开始批量处理")
        self.btn_start.clicked.connect(self.start_batch)
        left_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止")
        self.btn_stop.clicked.connect(self.stop_batch)
        self.btn_stop.setEnabled(False)
        left_layout.addWidget(self.btn_stop)

        self.progress = QProgressBar()
        left_layout.addWidget(self.progress)
        self.lbl_status = QLabel("就绪")
        left_layout.addWidget(self.lbl_status)

        self.preview_list = QListWidget()
        self.preview_list.itemClicked.connect(self.show_preview)
        left_layout.addWidget(QLabel("抽查结果"))
        left_layout.addWidget(self.preview_list)

        left_layout.addStretch()

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_view = ImageView()
        right_layout.addWidget(QLabel("抽查预览"))
        right_layout.addWidget(self.preview_view)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([350, 850])
        layout.addWidget(splitter)

    def select_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择输入文件夹")
        if path:
            self.input_dir = path
            self.lbl_input.setText(path)

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出文件夹")
        if path:
            self.output_dir = path
            self.lbl_output.setText(path)

    def start_batch(self):
        if self.preview_mode:
            self._continue_after_preview()
            return
        if not self.input_dir or not self.output_dir:
            QMessageBox.warning(self, "提示", "请先选择输入和输出文件夹")
            return

        self.image_paths = []
        for ext in SUPPORTED_IMAGE_EXTS:
            self.image_paths.extend(Path(self.input_dir).glob(f"**/{ext}"))
        self.image_paths = sorted(self.image_paths)

        if not self.image_paths:
            QMessageBox.warning(self, "提示", "输入文件夹中没有支持的图片")
            return

        reply = QMessageBox.question(
            self, "确认", "批量处理会按设置自动检测并去除水印。是否先抽查前 3 张？",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )
        if reply == QMessageBox.Cancel:
            return

        preview_count = 3 if reply == QMessageBox.Yes else 0
        self.preview_stop_index = min(preview_count - 1, len(self.image_paths) - 1) if preview_count > 0 else -1
        self.preview_list.clear()
        self.preview_queue.clear()
        self.preview_view.set_image(None)
        self.progress.setValue(0)
        self.progress.setMaximum(len(self.image_paths))
        self.lbl_status.setText("准备处理...")
        self.set_busy(True)

        if preview_count > 0:
            self.preview_mode = True

        self.batch_thread = BatchProcessThread(
            self.processor, self.image_paths, self.input_dir, self.output_dir,
            self.prompt_edit.text(), self.max_bbox_spin.value(),
            self.dilation_spin.value(),
            start_index=0,
            preview_count=preview_count,
            preview_stop_at=self.preview_stop_index
        )
        self.batch_thread.progress.connect(self.on_progress)
        self.batch_thread.preview.connect(self.on_preview)
        self.batch_thread.finished.connect(self.on_finished)
        self.batch_thread.error.connect(self.on_error)
        self.batch_thread.start()

    def on_progress(self, current, total, name):
        self.progress.setValue(current)
        self.lbl_status.setText(f"处理中 {current}/{total}: {name}")

    def on_preview(self, image, name):
        self.preview_queue.append((image, name))
        item = QListWidgetItem(name)
        item.setData(Qt.UserRole, len(self.preview_queue) - 1)
        self.preview_list.addItem(item)
        if len(self.preview_queue) == 1:
            self.show_preview(item)

    def show_preview(self, item):
        idx = item.data(Qt.UserRole)
        if 0 <= idx < len(self.preview_queue):
            image, name = self.preview_queue[idx]
            self.preview_view.set_image(image)
            self.lbl_status.setText(f"预览: {name}")

    def _continue_after_preview(self):
        self.preview_mode = False
        self.btn_start.setText("开始批量处理")
        self.set_busy(True)
        start = self.preview_stop_index + 1
        if start >= len(self.image_paths):
            self.on_finished(len(self.image_paths), len(self.image_paths))
            return
        self.batch_thread = BatchProcessThread(
            self.processor, self.image_paths, self.input_dir, self.output_dir,
            self.prompt_edit.text(), self.max_bbox_spin.value(),
            self.dilation_spin.value(),
            start_index=start,
            preview_count=0
        )
        self.batch_thread.progress.connect(self.on_progress)
        self.batch_thread.preview.connect(self.on_preview)
        self.batch_thread.finished.connect(self.on_finished)
        self.batch_thread.error.connect(self.on_error)
        self.batch_thread.start()

    def stop_batch(self):
        if self.batch_thread:
            self.batch_thread.stop()
            self.batch_thread.wait()
        self.set_busy(False)
        self.lbl_status.setText("已停止")

    def on_finished(self, processed, total):
        self.set_busy(False)
        self.lbl_status.setText(f"完成：{processed}/{total}")
        if not self.preview_mode:
            QMessageBox.information(self, "完成", f"批量处理完成，共处理 {processed} 张图片。")

    def on_error(self, msg):
        QMessageBox.critical(self, "错误", msg)

    def set_busy(self, busy):
        if self.preview_mode:
            self.btn_start.setText("继续处理剩余图片")
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)
        self.btn_input.setEnabled(not busy)
        self.btn_output.setEnabled(not busy)
