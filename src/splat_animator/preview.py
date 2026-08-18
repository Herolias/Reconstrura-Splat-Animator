from __future__ import annotations

import time

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFocusEvent,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import QLabel, QSizePolicy

from .theme import COLORS


class PreviewCanvas(QLabel):
    dragged = Signal(float, float)
    panned = Signal(float, float)
    zoomed = Signal(float)
    navigated = Signal(float, float, float, float)
    resetRequested = Signal()
    resized = Signal()

    _NAVIGATION_KEYS = {
        Qt.Key_W,
        Qt.Key_A,
        Qt.Key_S,
        Qt.Key_D,
        Qt.Key_Q,
        Qt.Key_E,
        Qt.Key_Shift,
    }

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("previewCanvas")
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(520, 340)
        # QLabel normally derives its size hint from its pixmap. Interactive
        # and settled previews intentionally have different pixel dimensions,
        # so the layout must ignore that changing image size.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(
            "Left drag: orbit\n"
            "Middle or right drag: pan\n"
            "Mouse wheel: zoom\n"
            "WASD and Q/E: move (hold Shift to move faster)\n"
            "Double-click: reset view"
        )
        self._image: QImage | None = None
        self._display_pixmap: QPixmap | None = None
        self._last_position: QPoint | None = None
        self._drag_mode: str | None = None
        self._pressed_keys: set[int] = set()
        self._navigation_timer = QTimer(self)
        self._navigation_timer.setInterval(30)
        self._navigation_timer.timeout.connect(self._emit_navigation)
        self._last_navigation_tick = time.monotonic()
        self._message = "Choose a project or splat file to begin"

    def set_message(self, message: str) -> None:
        self._message = message
        if self._image is None:
            self.update()

    def set_preview(self, image: QImage) -> None:
        self._image = image
        self._update_pixmap()

    def clear_preview(self, message: str) -> None:
        self._image = None
        self._display_pixmap = None
        self._message = message
        self.clear()
        self.update()

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(720, 480)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(520, 340)

    def _update_pixmap(self) -> None:
        if self._image is None:
            return
        margins = self.contentsMargins()
        target = QSize(
            max(1, self.width() - margins.left() - margins.right()),
            max(1, self.height() - margins.top() - margins.bottom()),
        )
        self._display_pixmap = QPixmap.fromImage(self._image).scaled(
            target,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.update()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_pixmap()
        self.resized.emit()

    def paintEvent(self, event: QEvent) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        if self._display_pixmap is not None:
            x = (self.width() - self._display_pixmap.width()) // 2
            y = (self.height() - self._display_pixmap.height()) // 2
            if self._display_pixmap.hasAlphaChannel():
                tile = 14
                width = self._display_pixmap.width()
                height = self._display_pixmap.height()
                painter.fillRect(x, y, width, height, QColor("#252b35"))
                for row in range(0, height, tile):
                    for column in range(0, width, tile):
                        if (row // tile + column // tile) % 2 == 0:
                            painter.fillRect(
                                x + column,
                                y + row,
                                min(tile, width - column),
                                min(tile, height - row),
                                QColor("#3a424f"),
                            )
            painter.drawPixmap(x, y, self._display_pixmap)
        elif self._image is None:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(QColor(COLORS["faint"]))
            painter.drawText(self.rect(), Qt.AlignCenter, self._message)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self.setFocus(Qt.MouseFocusReason)
            self._last_position = event.position().toPoint()
            shift_pan = bool(event.modifiers() & Qt.ShiftModifier)
            self._drag_mode = (
                "orbit" if event.button() == Qt.LeftButton and not shift_pan else "pan"
            )
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last_position is None:
            return
        current = event.position().toPoint()
        delta = current - self._last_position
        self._last_position = current
        if self._drag_mode == "pan":
            self.panned.emit(float(delta.x()), float(delta.y()))
        else:
            self.dragged.emit(float(delta.x()), float(delta.y()))
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() in (Qt.LeftButton, Qt.RightButton, Qt.MiddleButton):
            self._last_position = None
            self._drag_mode = None
            self.setCursor(Qt.OpenHandCursor)
            event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        self.setFocus(Qt.MouseFocusReason)
        self.zoomed.emit(float(event.angleDelta().y()))
        event.accept()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.resetRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key not in self._NAVIGATION_KEYS:
            super().keyPressEvent(event)
            return
        if not event.isAutoRepeat():
            self._pressed_keys.add(key)
            if not self._navigation_timer.isActive():
                self._last_navigation_tick = time.monotonic()
                self._navigation_timer.start()
        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key not in self._NAVIGATION_KEYS:
            super().keyReleaseEvent(event)
            return
        if not event.isAutoRepeat():
            self._pressed_keys.discard(key)
            if not self._pressed_keys:
                self._navigation_timer.stop()
        event.accept()

    def focusOutEvent(self, event: QFocusEvent) -> None:
        self._pressed_keys.clear()
        self._navigation_timer.stop()
        super().focusOutEvent(event)

    def _emit_navigation(self) -> None:
        now = time.monotonic()
        elapsed = min(0.08, max(0.0, now - self._last_navigation_tick))
        self._last_navigation_tick = now
        right = float((Qt.Key_D in self._pressed_keys) - (Qt.Key_A in self._pressed_keys))
        upward = float((Qt.Key_E in self._pressed_keys) - (Qt.Key_Q in self._pressed_keys))
        forward = float((Qt.Key_W in self._pressed_keys) - (Qt.Key_S in self._pressed_keys))
        speed = 3.0 if Qt.Key_Shift in self._pressed_keys else 1.0
        if right or upward or forward:
            self.navigated.emit(right * speed, upward * speed, forward * speed, elapsed)
