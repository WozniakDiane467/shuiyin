import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from .gui.main_window import MainWindow


def run():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("qu-shuiyin")
    app.setApplicationDisplayName("去水印小工具")

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run()
