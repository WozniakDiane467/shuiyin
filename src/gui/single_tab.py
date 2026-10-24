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


class ProcessSingleThread(QThread):
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(self, processor, image, regions, dilation):
        super().__init__()
        self.processor = processor
        self.image = image
        self.regions = regions
        self.dilation = dilation

    def run(self):
        try:
            result, mask = self.processor.process(self.image, self.regions, self.dilation)
            self.finished.emit(result, mask)
        except Exception as e:
            self.error.emit(str(e))


class DetectThread(QThread):
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, processor, image, prompt, max_bbox_percent):
        super().__init__()
        self.processor = processor
        self.image = image
        self.prompt = prompt
        self.max_bbox_percent = max_bbox_percent

    def run(self):
        try:
            regions = self.processor.detect(self.image, self.prompt, self.max_bbox_percent)
            self.finished.emit(regions)
        except Exception as e:
            self.error.emit(str(e))


class SingleTab(QWidget):
    def __init__(self, processor: ImageProcessor, parent=None):
        super().__init__(parent)
        self.processor = processor
        self.current_image = None
        self.current_result = None
        self.current_mask = None
        self.regions = []

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Left panel
        left = QWidget()
        left_layout = QVBoxLayout(left)

        self.btn_load = QPushButton("选择图片")
        self.btn_load.clicked.connect(self.load_image)
        left_layout.addWidget(self.btn_load)

        settings = QGroupBox("检测设置")
        settings_layout = QVBoxLayout(settings)
        settings_layout.addWidget(QLabel("检测提示词（用逗号分隔）"))
        self.prompt_edit = QLineEdit("watermark, logo, text")
        settings_layout.addWidget(self.prompt_edit)
        settings_layout.addWidget(QLabel("最大水印区域占比(%)"))
        self.max_bbox_spin = QSpinBox()
        self.max_bbox_spin.setRange(1, 50)
        self.max_bbox_spin.setValue(15)
        settings_layout.addWidget(self.max_bbox_spin)
        settings_layout.addWidget(QLabel("蒙版膨胀半径（像素）"))
        self.dilation_spin = QSpinBox()
        self.dilation_spin.setRange(0, 30)
        self.dilation_spin.setValue(5)
        settings_layout.addWidget(self.dilation_spin)
        left_layout.addWidget(settings)

        self.btn_detect = QPushButton("自动检测水印")
        self.btn_detect.clicked.connect(self.auto_detect)
        self.btn_detect.setEnabled(False)
        left_layout.addWidget(self.btn_detect)

        self.btn_draw = QPushButton("手动框选水印区域")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self.toggle_drawing)
        self.btn_draw.setEnabled(False)
        left_layout.addWidget(self.btn_draw)

        regions_group = QGroupBox("检测到的区域")
        regions_layout = QVBoxLayout(regions_group)
        self.region_list = QListWidget()
        self.region_list.itemSelectionChanged.connect(self.highlight_region)
        regions_layout.addWidget(self.region_list)
        self.btn_delete_region = QPushButton("删除选中区域")
        self.btn_delete_region.clicked.connect(self.delete_region)
        regions_layout.addWidget(self.btn_delete_region)
        left_layout.addWidget(regions_group)

        self.btn_process = QPushButton("处理图片")
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        left_layout.addWidget(self.btn_process)

        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        left_layout.addWidget(self.progress)

        left_layout.addStretch()

        # Right panel
        right = QWidget()
        right_layout = QVBoxLayout(right)
        tabs = QSplitter(Qt.Horizontal)

        self.original_view = ImageView()
        self.original_view.setToolTip("原图 / 蒙版预览")
        self.result_view = ImageView()
        self.result_view.setToolTip("处理结果")

        tabs.addWidget(self._wrap_view("原图", self.original_view))
        tabs.addWidget(self._wrap_view("结果", self.result_view))
        tabs.setSizes([400, 400])
        right_layout.addWidget(tabs)

        self.original_view.region_drawn.connect(self.add_manual_region)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

    def _wrap_view(self, title, view):
        widget = QWidget()
        l = QVBoxLayout(widget)
        l.addWidget(QLabel(title))
        l.addWidget(view)
        return widget

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if not path:
            return
        try:
            self.current_image = load_image(path)
            self.current_result = None
            self.current_mask = None
            self.regions = []
            self.region_list.clear()
            self.original_view.set_image(self.current_image)
            self.result_view.set_image(None)
            self.btn_detect.setEnabled(True)
            self.btn_draw.setEnabled(True)
            self.btn_process.setEnabled(True)
            self.btn_save.setEnabled(False)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败：{e}")

    def auto_detect(self):
        if self.current_image is None:
            return
        self.set_busy(True)
        prompt = self.prompt_edit.text()
        max_pct = self.max_bbox_spin.value()
        self.thread = DetectThread(self.processor, self.current_image, prompt, max_pct)
        self.thread.finished.connect(self.on_detect_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_detect_finished(self, regions):
        self.regions = regions
        self.refresh_region_list()
        self.original_view.set_regions(self.regions)
        self.set_busy(False)

    def on_error(self, msg):
        self.set_busy(False)
        QMessageBox.critical(self, "错误", msg)

    def toggle_drawing(self, checked):
        self.original_view.set_drawing_enabled(checked)
        self.btn_draw.setText("完成框选" if checked else "手动框选水印区域")

    def add_manual_region(self, rect):
        self.regions.append({
            "bbox": [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()],
            "label": "manual"
        })
        self.refresh_region_list()
        self.original_view.set_regions(self.regions)
        self.btn_draw.setChecked(False)

    def refresh_region_list(self):
        self.region_list.clear()
        for i, region in enumerate(self.regions):
            x1, y1, x2, y2 = region["bbox"]
            label = region.get("label", "unknown")
            item = QListWidgetItem(f"{i+1}. {label} ({x1},{y1}-{x2},{y2})")
            item.setData(Qt.UserRole, i)
            self.region_list.addItem(item)

    def highlight_region(self):
        selected = self.region_list.selectedItems()
        if not selected:
            return
        idx = selected[0].data(Qt.UserRole)
        if 0 <= idx < len(self.regions):
            # TODO: highlight single region
            pass

    def delete_region(self):
        selected = self.region_list.selectedItems()
        if not selected:
            return
        for item in selected:
            idx = item.data(Qt.UserRole)
            if 0 <= idx < len(self.regions):
                self.regions[idx] = None
        self.regions = [r for r in self.regions if r is not None]
        self.refresh_region_list()
        self.original_view.set_regions(self.regions)

    def process_image(self):
        if self.current_image is None or not self.regions:
            QMessageBox.warning(self, "提示", "请先加载图片并框选水印区域")
            return
        self.set_busy(True)
        dilation = self.dilation_spin.value()
        self.thread = ProcessSingleThread(self.processor, self.current_image, self.regions, dilation)
        self.thread.finished.connect(self.on_process_finished)
        self.thread.error.connect(self.on_error)
        self.thread.start()

    def on_process_finished(self, result, mask):
        self.current_result = result
        self.current_mask = mask
        self.result_view.set_image(result)
        self.btn_save.setEnabled(True)
        self.set_busy(False)

    def save_result(self):
        if self.current_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;WEBP (*.webp)"
        )
        if not path:
            return
        try:
            save_image(self.current_result, path)
            QMessageBox.information(self, "完成", f"已保存到：{path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败：{e}")

    def set_busy(self, busy):
        self.progress.setVisible(busy)
        self.btn_detect.setEnabled(not busy and self.current_image is not None)
        self.btn_process.setEnabled(not busy and self.current_image is not None)
        self.btn_load.setEnabled(not busy)
        self.btn_save.setEnabled(not busy and self.current_result is not None)
