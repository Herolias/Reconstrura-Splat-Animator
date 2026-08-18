from __future__ import annotations

import base64
from pathlib import Path

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap

APP_NAME = "Reconstrura Splat Animator"
_LOGO_PATH = Path(__file__).with_name("assets") / "reconstrura-logo.png.b64"


def app_icon(size: int = 64) -> QIcon:
    try:
        logo_data = base64.b64decode(_LOGO_PATH.read_text(encoding="ascii"))
        official = QPixmap()
        if official.loadFromData(logo_data, "PNG"):
            return QIcon(official)
    except (OSError, ValueError):
        pass

    # A package damaged during installation should still have a usable icon.
    canvas = QPixmap(size, size)
    canvas.fill(Qt.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(QColor("#3b465b"), max(1.0, size / 32)))
    painter.setBrush(QColor("#101722"))
    painter.drawRoundedRect(
        size * 0.08, size * 0.08, size * 0.84, size * 0.84, size * 0.2, size * 0.2
    )
    dots = (
        (0.29, 0.53, 0.10, "#f1f5ff"),
        (0.40, 0.36, 0.075, "#7ca4ff"),
        (0.53, 0.29, 0.055, "#a783ff"),
        (0.58, 0.48, 0.085, "#eaf0ff"),
        (0.67, 0.63, 0.05, "#6d8fff"),
        (0.49, 0.68, 0.065, "#aa7fff"),
        (0.72, 0.40, 0.03, "#e8ecff"),
    )
    painter.setPen(Qt.NoPen)
    for x, y, radius, color in dots:
        painter.setBrush(QColor(color))
        painter.drawEllipse(
            QPoint(round(size * x), round(size * y)), round(size * radius), round(size * radius)
        )
    painter.end()
    return QIcon(canvas)


def short_gpu_name(name: str) -> str:
    cleaned = name.split("/PCIe", 1)[0].strip()
    for prefix in (
        "NVIDIA GeForce ",
        "NVIDIA ",
        "AMD Radeon ",
        "AMD ",
        "Intel(R) ",
        "Intel ",
    ):
        if cleaned.startswith(prefix):
            return cleaned[len(prefix) :]
    return cleaned or "GPU"
