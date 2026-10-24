from PySide6.QtWidgets import QWidget, QLabel, QSizePolicy, QRubberBand
from PySide6.QtCore import Qt, QRect, QPoint, QSize, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PIL import Image


class ImageView(QWidget):
    region_drawn = Signal(QRect)
    region_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1a1a1a;")

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._pixmap = QPixmap()
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber_band.setStyleSheet("border: 2px dashed red;")

        self._start_pos = None
        self._regions = []
        self._drawing_enabled = False

    def set_image(self, image: Image.Image | QPixmap | None):
        if image is None:
            self._pixmap = QPixmap()
            self._label.clear()
            return
        if isinstance(image, Image.Image):
            image = image.convert("RGBA")
            data = image.tobytes("raw", "RGBA")
            qimage = QImage(data, image.width, image.height, QImage.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
        else:
            pixmap = image
        self._pixmap = pixmap
        self._update_display()

    def set_regions(self, regions: list):
        self._regions = regions
        self._update_display()

    def set_drawing_enabled(self, enabled: bool):
        self._drawing_enabled = enabled

    def _update_display(self):
        if self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = scaled.width() / self._pixmap.width()
        self._offset = QPoint((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2)

        canvas = QPixmap(self.size())
        canvas.fill(QColor("#1a1a1a"))
        painter = QPainter(canvas)
        painter.drawPixmap(self._offset, scaled)

        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        for i, region in enumerate(self._regions):
            x1, y1, x2, y2 = region["bbox"]
            rx = int(x1 * self._scale) + self._offset.x()
            ry = int(y1 * self._scale) + self._offset.y()
            rw = int((x2 - x1) * self._scale)
            rh = int((y2 - y1) * self._scale)
            painter.drawRect(rx, ry, rw, rh)
            painter.drawText(rx + 2, ry - 4, str(i + 1))

        painter.end()
        self._label.setPixmap(canvas)
        self._label.setGeometry(self.rect())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def mousePressEvent(self, event):
        if self._drawing_enabled and event.button() == Qt.LeftButton:
            self._start_pos = event.pos()
            self._rubber_band.setGeometry(QRect(self._start_pos, QSize()))
            self._rubber_band.show()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing_enabled and self._start_pos:
            rect = QRect(self._start_pos, event.pos()).normalized()
            self._rubber_band.setGeometry(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drawing_enabled and event.button() == Qt.LeftButton and self._start_pos:
            rect = QRect(self._start_pos, event.pos()).normalized()
            self._rubber_band.hide()
            self._start_pos = None
            if rect.width() > 5 and rect.height() > 5:
                mapped = self._map_to_image(rect)
                if mapped.isValid():
                    self.region_drawn.emit(mapped)
        super().mouseReleaseEvent(event)

    def _map_to_image(self, rect: QRect) -> QRect:
        x1 = int((rect.x() - self._offset.x()) / self._scale)
        y1 = int((rect.y() - self._offset.y()) / self._scale)
        x2 = int((rect.right() - self._offset.x()) / self._scale)
        y2 = int((rect.bottom() - self._offset.y()) / self._scale)
        x1 = max(0, min(x1, self._pixmap.width() - 1))
        y1 = max(0, min(y1, self._pixmap.height() - 1))
        x2 = max(0, min(x2, self._pixmap.width() - 1))
        y2 = max(0, min(y2, self._pixmap.height() - 1))
        return QRect(x1, y1, x2 - x1, y2 - y1).normalized()
