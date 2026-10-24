from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar, QMessageBox
from PySide6.QtCore import Qt

from .single_tab import SingleTab
from .batch_tab import BatchTab
from ..core.processor import ImageProcessor
from ..config import ensure_dirs


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.setWindowTitle("去水印小工具 - qu-shuiyin")
        self.setMinimumSize(1200, 800)

        self.processor = ImageProcessor()

        self.tabs = QTabWidget()
        self.tabs.addTab(SingleTab(self.processor), "单张处理")
        self.tabs.addTab(BatchTab(self.processor), "批量处理")
        self.setCentralWidget(self.tabs)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("就绪")

    def closeEvent(self, event):
        event.accept()
