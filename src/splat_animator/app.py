from __future__ import annotations

import argparse
import base64
import shutil
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEvent,
    QObject,
    QPoint,
    QProcess,
    QSize,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QFocusEvent,
    QIcon,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QResizeEvent,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .io import ProjectDetection, detect_project, inspect_source, load_scene
from .models import AnimationSettings
from .renderer import (
    GpuRenderer,
    RenderCancelled,
    camera_basis,
    camera_target_offset,
    expected_extension,
    render_video,
)
from .theme import COLORS, apply_theme

APP_NAME = "Reconstrura Splat Animator"
_LOGO_PATH = Path(__file__).with_name("assets") / "reconstrura-logo.png.b64"


def _app_icon(size: int = 64) -> QIcon:
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


def _short_gpu_name(name: str) -> str:
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


class RenderWorker(QObject):
    sceneLoaded = Signal(object)
    previewReady = Signal(QImage)
    progress = Signal(int, str)
    renderFinished = Signal(str)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.renderer: GpuRenderer | None = None
        self.cancel_event = threading.Event()
        self.rendering = False

    @Slot(str, int)
    def load(self, path: str, budget: int) -> None:
        if self.rendering:
            return
        try:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            scene = load_scene(path, None if budget <= 0 else budget)
            self.renderer = GpuRenderer(scene)
            self.sceneLoaded.emit(
                {
                    "description": scene.source_info.description,
                    "loaded": scene.count,
                    "original": scene.original_count,
                    "radius": scene.radius,
                    "renderer": self.renderer.renderer_name,
                    "path": str(scene.source_info.path),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))

    @Slot(object, float, int, int, bool)
    def preview(
        self,
        settings: AnimationSettings,
        seconds: float,
        width: int,
        height: int,
        sort_depth: bool,
    ) -> None:
        if self.renderer is None or self.rendering:
            return
        try:
            width = max(16, width - width % 2)
            height = max(16, height - height % 2)
            preview_settings = replace(settings, width=width, height=height)
            data = self.renderer.render_rgb(
                preview_settings,
                seconds,
                sort_depth=sort_depth,
            )
            array = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)[::-1].copy()
            image = QImage(array.data, width, height, width * 3, QImage.Format_RGB888).copy()
            self.previewReady.emit(image)
        except Exception as exc:
            self.failed.emit(f"Preview failed: {exc}")

    @Slot(object, str)
    def render(self, settings: AnimationSettings, output: str) -> None:
        if self.renderer is None or self.rendering:
            return
        self.rendering = True
        self.cancel_event.clear()
        try:

            def on_progress(done: int, total: int) -> None:
                percent = round(done * 100 / total)
                self.progress.emit(percent, f"Rendering frame {done:,} of {total:,}")

            path = render_video(
                self.renderer,
                settings,
                output,
                progress=on_progress,
                cancel_event=self.cancel_event,
            )
            self.renderFinished.emit(str(path))
        except RenderCancelled:
            self.failed.emit("Render cancelled. No partial video was kept.")
        except Exception as exc:
            self.failed.emit(f"Render failed: {exc}")
        finally:
            self.rendering = False

    def cancel(self) -> None:
        self.cancel_event.set()

    @Slot()
    def shutdown(self) -> None:
        self.cancel_event.set()
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.stopped.emit()
        QThread.currentThread().quit()


class MainWindow(QMainWindow):
    requestLoad = Signal(str, int)
    requestPreview = Signal(object, float, int, int, bool)
    requestRender = Signal(object, str)
    requestShutdown = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(_app_icon())
        self.setMinimumSize(1160, 720)
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1500, 930)
        else:
            available = screen.availableGeometry()
            self.resize(
                max(self.minimumWidth(), min(1500, available.width() - 32)),
                max(self.minimumHeight(), min(930, available.height() - 32)),
            )
        self.detection: ProjectDetection | None = None
        self.scene_loaded = False
        self.render_active = False
        self.preview_inflight = False
        self.preview_pending = False
        self._preview_fast = False
        self._last_preview_target_size: tuple[int, int] | None = None
        defaults = AnimationSettings()
        self._background = defaults.background
        self._min_splat_pixels = defaults.min_splat_pixels
        self._max_splat_pixels = defaults.max_splat_pixels
        self._play_seconds = 0.0
        self._last_tick = time.monotonic()
        self._renderer_name = ""
        self._nvidia_smi = shutil.which("nvidia-smi")

        self.worker_thread = QThread(self)
        self.worker_thread.setObjectName("SplatRenderer")
        self.worker = RenderWorker()
        self.worker.moveToThread(self.worker_thread)
        self.requestLoad.connect(self.worker.load, Qt.QueuedConnection)
        self.requestPreview.connect(self.worker.preview, Qt.QueuedConnection)
        self.requestRender.connect(self.worker.render, Qt.QueuedConnection)
        self.requestShutdown.connect(self.worker.shutdown, Qt.QueuedConnection)
        self.worker.sceneLoaded.connect(self._scene_loaded, Qt.QueuedConnection)
        self.worker.previewReady.connect(self._preview_ready, Qt.QueuedConnection)
        self.worker.progress.connect(self._render_progress, Qt.QueuedConnection)
        self.worker.renderFinished.connect(self._render_finished, Qt.QueuedConnection)
        self.worker.failed.connect(self._worker_failed, Qt.QueuedConnection)
        self.worker_thread.start()

        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(90)
        self.preview_timer.timeout.connect(self._request_preview)
        self.preview_settle_timer = QTimer(self)
        self.preview_settle_timer.setSingleShot(True)
        self.preview_settle_timer.setInterval(180)
        self.preview_settle_timer.timeout.connect(self._preview_interaction_settled)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(33)
        self.play_timer.timeout.connect(self._play_tick)

        self._build_ui()
        self.gpu_process = QProcess(self)
        self.gpu_process.finished.connect(self._gpu_query_finished)
        self.gpu_timer = QTimer(self)
        self.gpu_timer.setInterval(1500)
        self.gpu_timer.timeout.connect(self._request_gpu_update)
        self.gpu_timer.start()
        QTimer.singleShot(0, self._request_gpu_update)
        self._connect_changes()
        self._apply_settings(AnimationSettings())
        self._update_timeline_summary()

    @staticmethod
    def _label(text: str, object_name: str | None = None, *, muted: bool = False) -> QLabel:
        label = QLabel(text)
        if object_name:
            label.setObjectName(object_name)
        if muted:
            label.setProperty("muted", True)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _combo(items: list[tuple[str, object]]) -> QComboBox:
        combo = QComboBox()
        combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(6)
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for label, data in items:
            combo.addItem(label, data)
        return combo

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        *,
        decimals: int = 2,
        step: float = 0.1,
        suffix: str = "",
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(0)
        spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        return spin

    @staticmethod
    def _spin(
        minimum: int, maximum: int, value: int, *, step: int = 1, suffix: str = ""
    ) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(0)
        spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        return spin

    def _card(self, title: str, subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("card")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(17, 15, 17, 17)
        layout.setSpacing(10)
        layout.addWidget(self._label(title.upper(), "sectionLabel"))
        if subtitle:
            layout.addWidget(self._label(subtitle, "hint"))
        return frame, layout

    def _field(self, layout: QVBoxLayout, title: str, widget: QWidget, hint: str = "") -> None:
        layout.addWidget(self._label(title, "fieldLabel"))
        layout.addWidget(widget)
        if hint:
            layout.addWidget(self._label(hint, "hint"))

    @staticmethod
    def _path_row(line: QLineEdit, button: QPushButton) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        layout.addWidget(line, 1)
        layout.addWidget(button)
        return row

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._build_rail(outer)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(24, 18, 18, 16)
        workspace_layout.setSpacing(14)

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(2)
        header_text.addWidget(self._label("Splat animation", "pageTitle"))
        header.addLayout(header_text)
        header.addStretch(1)
        self.load_preset_button = QPushButton("Load preset")
        self.save_preset_button = QPushButton("Save preset")
        header.addWidget(self.load_preset_button)
        header.addWidget(self.save_preset_button)
        workspace_layout.addLayout(header)

        preview_frame = QFrame()
        preview_frame.setObjectName("previewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(1, 1, 1, 10)
        preview_layout.setSpacing(7)
        self.preview = PreviewCanvas()
        preview_layout.addWidget(self.preview, 1)

        player = QHBoxLayout()
        player.setContentsMargins(12, 0, 12, 0)
        self.play_button = QPushButton("▶")
        self.play_button.setFixedWidth(42)
        self.play_button.setToolTip("Play or pause the preview timeline")
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setRange(0, 10_000)
        self.time_label = QLabel("0:00.0 / 0:12.0")
        self.time_label.setFixedWidth(110)
        player.addWidget(self.play_button)
        player.addWidget(self.timeline, 1)
        player.addWidget(self.time_label)
        preview_layout.addLayout(player)
        navigation_hint = self._label(
            "Left drag to orbit | Right or middle drag to pan | Scroll to zoom | "
            "WASD and Q/E to move",
            "hint",
        )
        navigation_hint.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(navigation_hint)
        workspace_layout.addWidget(preview_frame, 1)

        status_row = QHBoxLayout()
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {COLORS['blue']};")
        self.status_label = self._label("Choose a source to begin", muted=True)
        self.loaded_label = self._label("No scene loaded", "hint")
        # Preview progress text must never alter the row height: doing so
        # resizes the canvas, which itself schedules another preview.
        self.status_label.setWordWrap(False)
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.loaded_label.setWordWrap(False)
        self.loaded_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        status_row.addWidget(self.status_dot)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.loaded_label)
        workspace_layout.addLayout(status_row)
        outer.addWidget(workspace, 1)

        self._build_inspector(outer)

    def _build_rail(self, outer: QHBoxLayout) -> None:
        rail = QWidget()
        rail.setObjectName("rail")
        rail.setFixedWidth(218)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(14, 20, 14, 13)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        icon = QLabel()
        icon.setPixmap(_app_icon(38).pixmap(38, 38))
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        brand_text.addWidget(self._label("Reconstrura", "appName"))
        brand_text.addWidget(self._label("SPLAT ANIMATOR", "appTag"))
        brand.addWidget(icon)
        brand.addLayout(brand_text)
        brand.addStretch(1)
        layout.addLayout(brand)
        layout.addSpacing(25)

        layout.addWidget(self._label("SECTIONS", "sectionLabel"))
        self.nav_source = self._nav_button("Source")
        self.nav_motion = self._nav_button("Animation")
        self.nav_look = self._nav_button("Camera && appearance")
        self.nav_export = self._nav_button("Export")
        for button in (self.nav_source, self.nav_motion, self.nav_look, self.nav_export):
            layout.addWidget(button)
        self.nav_source.setChecked(True)
        layout.addStretch(1)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {COLORS['line']};")
        layout.addWidget(line)
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(6)
        self.gpu_name_label = self._label("GPU", "fieldLabel")
        self.gpu_usage_label = self._label("Detecting...", "hint")
        self.gpu_usage_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        gpu_row.addWidget(self.gpu_name_label, 1)
        gpu_row.addWidget(self.gpu_usage_label)
        layout.addLayout(gpu_row)
        self.gpu_meter = QProgressBar()
        self.gpu_meter.setObjectName("gpuMeter")
        self.gpu_meter.setRange(0, 1000)
        self.gpu_meter.setValue(0)
        self.gpu_meter.setTextVisible(False)
        self.gpu_meter.setFixedHeight(4)
        layout.addWidget(self.gpu_meter)
        outer.addWidget(rail)

    def _nav_button(self, text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("navButton")
        button.setCheckable(True)
        button.setAutoExclusive(True)
        return button

    def _build_inspector(self, outer: QHBoxLayout) -> None:
        self.inspector = QScrollArea()
        self.inspector.setWidgetResizable(True)
        self.inspector.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inspector.setFixedWidth(438)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.inspector_layout = QVBoxLayout(content)
        self.inspector_layout.setContentsMargins(10, 18, 17, 18)
        self.inspector_layout.setSpacing(12)
        self.inspector.setWidget(content)

        self._build_source_card()
        self._build_motion_card()
        self._build_camera_card()
        self._build_look_card()
        self._build_output_card()
        self.inspector_layout.addStretch(1)
        outer.addWidget(self.inspector)

        self.nav_source.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.source_card, 0, 12)
        )
        self.nav_motion.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.motion_card, 0, 12)
        )
        self.nav_look.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.camera_card, 0, 12)
        )
        self.nav_export.clicked.connect(
            lambda: self.inspector.ensureWidgetVisible(self.output_card, 0, 12)
        )

    def _build_source_card(self) -> None:
        self.source_card, layout = self._card("Source")
        self.project_line = QLineEdit()
        self.project_line.setPlaceholderText("Reconstrura or 3DGS project folder")
        self.project_browse = QPushButton("Browse")
        self.project_scan = QPushButton("Find files")
        project_row = QWidget()
        project_layout = QHBoxLayout(project_row)
        project_layout.setContentsMargins(0, 0, 0, 0)
        project_layout.setSpacing(7)
        project_layout.addWidget(self.project_line, 1)
        project_layout.addWidget(self.project_browse)
        project_layout.addWidget(self.project_scan)
        self._field(layout, "Project folder", project_row)

        self.candidate_combo = QComboBox()
        self.candidate_combo.setPlaceholderText("No files found")
        self._field(layout, "Detected file", self.candidate_combo)

        self.source_line = QLineEdit()
        self.source_line.setPlaceholderText(".ply or .splat")
        self.source_browse = QPushButton("Choose file")
        self._field(layout, "Splat file", self._path_row(self.source_line, self.source_browse))

        self.budget_spin = self._spin(0, 5_000_000, 0, step=50_000)
        self.budget_spin.setSpecialValueText("No limit")
        self._field(
            layout,
            "Point limit",
            self.budget_spin,
            "Use 0 for the full scene. A lower limit speeds up previews.",
        )
        self.source_info = self._label("No file selected", "hint")
        layout.addWidget(self.source_info)
        self.load_scene_button = QPushButton("Load file")
        self.load_scene_button.setProperty("primary", True)
        layout.addWidget(self.load_scene_button)
        self.inspector_layout.addWidget(self.source_card)

    def _build_motion_card(self) -> None:
        self.motion_card, layout = self._card("Animation")
        self.duration_spin = self._double_spin(0.5, 300.0, 12.0, suffix=" s", step=0.5)
        self._field(layout, "Duration", self.duration_spin)

        self.loop_toggle = QCheckBox("Seamless loop")
        self._field(
            layout,
            "Looping",
            self.loop_toggle,
            (
                "Adjusts the duration and spin speed to complete a full turn. "
                "Transitions return to the starting view."
            ),
        )

        self.start_representation = self._combo(
            [("Gaussian splat", "splat"), ("Point cloud", "point")]
        )
        self._field(layout, "Start with", self.start_representation)

        self.trip_mode = self._combo(
            [
                ("Return to start", True),
                ("One-way", False),
                ("Off", None),
            ]
        )
        self._field(layout, "Transition", self.trip_mode)

        timing_row = QWidget()
        timing_layout = QHBoxLayout(timing_row)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        timing_layout.setSpacing(7)
        self.transition_start_spin = self._double_spin(0.0, 300.0, 2.0, suffix=" s")
        self.transition_duration_spin = self._double_spin(0.1, 120.0, 2.2, suffix=" s")
        timing_layout.addWidget(self.transition_start_spin)
        timing_layout.addWidget(self.transition_duration_spin)
        self._field(
            layout,
            "Start time / duration",
            timing_row,
        )

        self.return_hold_spin = self._double_spin(0.0, 120.0, 2.6, suffix=" s")
        self._field(layout, "Hold before return", self.return_hold_spin)

        self.easing_combo = self._combo(
            [("Ease in/out", "cinematic"), ("Smooth", "smooth"), ("Linear", "linear")]
        )
        self._field(layout, "Easing", self.easing_combo)

        self.transition_effect = self._combo(
            [
                ("Sweep", "sweep"),
                ("Radial", "radial"),
                ("Wave", "wave"),
                ("Spiral", "spiral"),
                ("Dissolve", "dissolve"),
            ]
        )
        self._field(layout, "Effect", self.transition_effect)

        scan_row = QWidget()
        scan_layout = QHBoxLayout(scan_row)
        scan_layout.setContentsMargins(0, 0, 0, 0)
        scan_layout.setSpacing(7)
        self.scan_direction = self._combo(
            [
                ("Top to bottom", "top_to_bottom"),
                ("Bottom to top", "bottom_to_top"),
            ]
        )
        self.up_axis = self._combo(
            [
                ("-Y up", "-y"),
                ("Y up", "y"),
                ("Z up", "z"),
                ("-Z up", "-z"),
                ("X up", "x"),
                ("-X up", "-x"),
            ]
        )
        scan_layout.addWidget(self.scan_direction, 2)
        scan_layout.addWidget(self.up_axis, 1)
        self._field(layout, "Direction / up axis", scan_row)

        self.scan_feather_spin = self._double_spin(0.005, 0.35, 0.055, decimals=3, step=0.01)
        self._field(layout, "Edge softness", self.scan_feather_spin)
        self.timeline_summary = self._label("", "hint")
        layout.addWidget(self.timeline_summary)
        self.inspector_layout.addWidget(self.motion_card)

    def _build_camera_card(self) -> None:
        self.camera_card, layout = self._card("Camera")
        self.rotation_mode = self._combo(
            [("Orbit camera", "camera_orbit"), ("Spin object", "object_spin")]
        )
        self._field(layout, "Rotation mode", self.rotation_mode)

        spin_row = QWidget()
        spin_layout = QHBoxLayout(spin_row)
        spin_layout.setContentsMargins(0, 0, 0, 0)
        spin_layout.setSpacing(7)
        self.spin_speed = self._double_spin(0.0, 720.0, 30.0, suffix="°/s", step=5.0)
        self.spin_direction = self._combo(
            [("Clockwise", "clockwise"), ("Counterclockwise", "counter_clockwise")]
        )
        spin_layout.addWidget(self.spin_speed)
        spin_layout.addWidget(self.spin_direction)
        self._field(layout, "Spin speed / direction", spin_row)

        angle_row = QWidget()
        angle_layout = QHBoxLayout(angle_row)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.setSpacing(7)
        self.start_angle = self._double_spin(-360.0, 360.0, 0.0, suffix="°", step=5.0)
        self.elevation = self._double_spin(-80.0, 80.0, 12.0, suffix="°", step=2.0)
        angle_layout.addWidget(self.start_angle)
        angle_layout.addWidget(self.elevation)
        self._field(layout, "Start angle / elevation", angle_row)

        camera_row = QWidget()
        camera_layout = QHBoxLayout(camera_row)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_layout.setSpacing(7)
        self.distance_scale = self._double_spin(
            0.0001, 100.0, 2.65, suffix="×", decimals=6, step=0.1
        )
        self.fov = self._double_spin(1.0, 140.0, 48.0, suffix="°", step=2.0)
        camera_layout.addWidget(self.distance_scale)
        camera_layout.addWidget(self.fov)
        self._field(layout, "Distance / field of view", camera_row)

        pan_row = QWidget()
        pan_layout = QHBoxLayout(pan_row)
        pan_layout.setContentsMargins(0, 0, 0, 0)
        pan_layout.setSpacing(7)
        self.camera_target_x = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.camera_target_y = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.camera_target_z = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        pan_layout.addWidget(self.camera_target_x, 1)
        pan_layout.addWidget(self.camera_target_y, 1)
        pan_layout.addWidget(self.camera_target_z, 1)
        self._field(
            layout,
            "Orbit target X / Y / Z",
            pan_row,
            "Offsets use the scene radius as one unit.",
        )

        self.rotation_center_mode = self._combo(
            [
                ("Automatic", "automatic"),
                ("Scene center", "scene"),
                ("Orbit target", "target"),
                ("Custom", "custom"),
            ]
        )
        self._field(layout, "Rotation center", self.rotation_center_mode)

        rotation_center_row = QWidget()
        rotation_center_layout = QHBoxLayout(rotation_center_row)
        rotation_center_layout.setContentsMargins(0, 0, 0, 0)
        rotation_center_layout.setSpacing(7)
        self.rotation_center_x = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.rotation_center_y = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        self.rotation_center_z = self._double_spin(
            -1000.0, 1000.0, 0.0, suffix="×", decimals=8, step=0.05
        )
        rotation_center_layout.addWidget(self.rotation_center_x, 1)
        rotation_center_layout.addWidget(self.rotation_center_y, 1)
        rotation_center_layout.addWidget(self.rotation_center_z, 1)
        self._field(layout, "Custom center X / Y / Z", rotation_center_row)

        camera_actions = QWidget()
        camera_actions_layout = QHBoxLayout(camera_actions)
        camera_actions_layout.setContentsMargins(0, 0, 0, 0)
        camera_actions_layout.setSpacing(7)
        self.opening_frame_button = QPushButton("Go to start")
        self.reset_camera_button = QPushButton("Reset view")
        camera_actions_layout.addWidget(self.opening_frame_button)
        camera_actions_layout.addWidget(self.reset_camera_button)
        layout.addWidget(camera_actions)
        self.inspector_layout.addWidget(self.camera_card)

    def _build_look_card(self) -> None:
        self.look_card, layout = self._card("Appearance")
        point_row = QWidget()
        point_layout = QHBoxLayout(point_row)
        point_layout.setContentsMargins(0, 0, 0, 0)
        point_layout.setSpacing(7)
        self.point_radius = self._double_spin(0.25, 12.0, 0.75, suffix=" px", step=0.1)
        self.point_opacity = self._double_spin(0.01, 1.0, 0.55, step=0.05)
        point_layout.addWidget(self.point_radius)
        point_layout.addWidget(self.point_opacity)
        self._field(
            layout,
            "Point size / opacity",
            point_row,
            "Point size is based on 1080p and scales with the output resolution.",
        )

        splat_row = QWidget()
        splat_layout = QHBoxLayout(splat_row)
        splat_layout.setContentsMargins(0, 0, 0, 0)
        splat_layout.setSpacing(7)
        self.splat_scale = self._double_spin(0.1, 5.0, 1.0, suffix="×", step=0.05)
        self.splat_opacity = self._double_spin(0.01, 2.0, 1.0, suffix="×", step=0.05)
        splat_layout.addWidget(self.splat_scale)
        splat_layout.addWidget(self.splat_opacity)
        self._field(layout, "Splat scale / opacity", splat_row)

        self.exposure = self._double_spin(-4.0, 4.0, 0.0, suffix=" EV", step=0.1)
        self._field(layout, "Exposure", self.exposure)

        background_row = QWidget()
        background_layout = QHBoxLayout(background_row)
        background_layout.setContentsMargins(0, 0, 0, 0)
        background_layout.setSpacing(7)
        self.background_button = QPushButton("Choose color")
        self.gradient = self._double_spin(0.0, 0.75, 0.16, step=0.05)
        background_layout.addWidget(self.background_button, 1)
        background_layout.addWidget(self.gradient)
        self._field(layout, "Background color / gradient", background_row)
        self.inspector_layout.addWidget(self.look_card)

    def _build_output_card(self) -> None:
        self.output_card, layout = self._card("Export")
        self.resolution_preset = self._combo(
            [
                ("Full HD (1920 × 1080)", (1920, 1080)),
                ("4K UHD (3840 × 2160)", (3840, 2160)),
                ("Square (1080 × 1080)", (1080, 1080)),
                ("Vertical (1080 × 1920)", (1080, 1920)),
                ("Vertical 4K (2160 × 3840)", (2160, 3840)),
                ("Custom", None),
            ]
        )
        self._field(layout, "Resolution preset", self.resolution_preset)

        resolution_row = QWidget()
        resolution_layout = QHBoxLayout(resolution_row)
        resolution_layout.setContentsMargins(0, 0, 0, 0)
        resolution_layout.setSpacing(7)
        self.width_spin = self._spin(16, 7680, 1920, step=2)
        self.height_spin = self._spin(16, 7680, 1080, step=2)
        self.fps_spin = self._spin(1, 240, 30, suffix=" fps")
        resolution_layout.addWidget(self.width_spin)
        resolution_layout.addWidget(self.height_spin)
        resolution_layout.addWidget(self.fps_spin)
        self._field(layout, "Width / height / frame rate", resolution_row)

        codec_row = QWidget()
        codec_layout = QHBoxLayout(codec_row)
        codec_layout.setContentsMargins(0, 0, 0, 0)
        codec_layout.setSpacing(7)
        self.codec_combo = self._combo(
            [
                ("H.264 (MP4)", "h264"),
                ("H.265 (MP4)", "h265"),
                ("ProRes 422 HQ (MOV)", "prores"),
                ("VP9 (WebM)", "vp9"),
            ]
        )
        self.quality_spin = self._spin(0, 51, 18)
        codec_layout.addWidget(self.codec_combo, 2)
        codec_layout.addWidget(self.quality_spin, 1)
        self._field(
            layout,
            "Format / quality (CRF)",
            codec_row,
            "Lower CRF means higher quality and larger files.",
        )

        self.output_line = QLineEdit()
        self.output_line.setPlaceholderText("promo.mp4")
        self.output_browse = QPushButton("Browse")
        self._field(layout, "Output file", self._path_row(self.output_line, self.output_browse))

        self.render_estimate = self._label("360 frames", "hint")
        layout.addWidget(self.render_estimate)
        self.render_button = QPushButton("Render video")
        self.render_button.setProperty("primary", True)
        self.render_button.setEnabled(False)
        layout.addWidget(self.render_button)
        self.cancel_button = QPushButton("Cancel render")
        self.cancel_button.setEnabled(False)
        layout.addWidget(self.cancel_button)
        self.render_progress = QProgressBar()
        self.render_progress.setRange(0, 100)
        self.render_progress.setValue(0)
        self.render_progress.setTextVisible(False)
        layout.addWidget(self.render_progress)
        self.render_status = self._label("Ready", "hint")
        layout.addWidget(self.render_status)
        self.inspector_layout.addWidget(self.output_card)

    def _connect_changes(self) -> None:
        self.project_browse.clicked.connect(self._choose_project)
        self.project_scan.clicked.connect(lambda: self._scan_project(False))
        self.project_line.returnPressed.connect(lambda: self._scan_project(False))
        self.source_browse.clicked.connect(self._choose_source)
        self.candidate_combo.currentIndexChanged.connect(self._candidate_changed)
        self.source_line.editingFinished.connect(self._inspect_manual_source)
        self.load_scene_button.clicked.connect(self._load_scene)
        self.output_browse.clicked.connect(self._choose_output)
        self.background_button.clicked.connect(self._choose_background)
        self.render_button.clicked.connect(self._start_render)
        self.cancel_button.clicked.connect(self._cancel_render)
        self.load_preset_button.clicked.connect(self._load_preset)
        self.save_preset_button.clicked.connect(self._save_preset)
        self.play_button.clicked.connect(self._toggle_play)
        self.timeline.valueChanged.connect(self._timeline_changed)
        self.timeline.sliderPressed.connect(self._preview_interaction_started)
        self.timeline.sliderReleased.connect(self._preview_interaction_settled)
        self.preview.dragged.connect(self._preview_dragged)
        self.preview.panned.connect(self._preview_panned)
        self.preview.zoomed.connect(self._preview_zoomed)
        self.preview.navigated.connect(self._preview_navigated)
        self.preview.resetRequested.connect(self._reset_camera)
        self.preview.resized.connect(self._preview_resized)
        self.opening_frame_button.clicked.connect(self._show_opening_frame)
        self.reset_camera_button.clicked.connect(self._reset_camera)
        self.resolution_preset.currentIndexChanged.connect(self._resolution_preset_changed)
        self.codec_combo.currentIndexChanged.connect(self._codec_changed)
        self.trip_mode.currentIndexChanged.connect(self._trip_mode_changed)
        self.loop_toggle.toggled.connect(self._loop_toggled)
        self.rotation_center_mode.currentIndexChanged.connect(self._rotation_center_mode_changed)

        change_widgets = (
            self.duration_spin,
            self.transition_start_spin,
            self.transition_duration_spin,
            self.return_hold_spin,
            self.scan_feather_spin,
            self.spin_speed,
            self.start_angle,
            self.elevation,
            self.distance_scale,
            self.camera_target_x,
            self.camera_target_y,
            self.camera_target_z,
            self.rotation_center_x,
            self.rotation_center_y,
            self.rotation_center_z,
            self.fov,
            self.point_radius,
            self.point_opacity,
            self.splat_scale,
            self.splat_opacity,
            self.exposure,
            self.gradient,
            self.width_spin,
            self.height_spin,
            self.fps_spin,
            self.quality_spin,
        )
        for widget in change_widgets:
            widget.valueChanged.connect(self._settings_changed)
        combo_widgets = (
            self.start_representation,
            self.trip_mode,
            self.easing_combo,
            self.transition_effect,
            self.scan_direction,
            self.up_axis,
            self.rotation_mode,
            self.spin_direction,
            self.codec_combo,
        )
        for widget in combo_widgets:
            widget.currentIndexChanged.connect(self._settings_changed)

    def _settings(self, *, resolve_loop: bool = True) -> AnimationSettings:
        width = self.width_spin.value() - self.width_spin.value() % 2
        height = self.height_spin.value() - self.height_spin.value() % 2
        settings = AnimationSettings(
            duration=self.duration_spin.value(),
            fps=self.fps_spin.value(),
            width=width,
            height=height,
            seamless_loop=self.loop_toggle.isChecked(),
            rotation_mode=str(self.rotation_mode.currentData()),
            spin_speed=self.spin_speed.value(),
            spin_direction=str(self.spin_direction.currentData()),
            start_angle=self.start_angle.value(),
            elevation=self.elevation.value(),
            distance_scale=self.distance_scale.value(),
            camera_target_x=self.camera_target_x.value(),
            camera_target_y=self.camera_target_y.value(),
            camera_target_z=self.camera_target_z.value(),
            rotation_center_mode=str(self.rotation_center_mode.currentData()),
            rotation_center_x=self.rotation_center_x.value(),
            rotation_center_y=self.rotation_center_y.value(),
            rotation_center_z=self.rotation_center_z.value(),
            fov=self.fov.value(),
            start_representation=str(self.start_representation.currentData()),
            transformation_enabled=self.trip_mode.currentData() is not None,
            round_trip=self.trip_mode.currentData() is True,
            transition_start=self.transition_start_spin.value(),
            transition_duration=self.transition_duration_spin.value(),
            return_hold=self.return_hold_spin.value(),
            easing=str(self.easing_combo.currentData()),
            transition_effect=str(self.transition_effect.currentData()),
            scan_direction=str(self.scan_direction.currentData()),
            up_axis=str(self.up_axis.currentData()),
            scan_feather=self.scan_feather_spin.value(),
            point_radius=self.point_radius.value(),
            point_opacity=self.point_opacity.value(),
            splat_scale=self.splat_scale.value(),
            splat_opacity=self.splat_opacity.value(),
            min_splat_pixels=self._min_splat_pixels,
            max_splat_pixels=self._max_splat_pixels,
            exposure=self.exposure.value(),
            background=self._background,
            background_gradient=self.gradient.value(),
            codec=str(self.codec_combo.currentData()),
            quality=self.quality_spin.value(),
        )
        return settings.resolved_for_loop() if resolve_loop else settings

    @staticmethod
    def _set_combo(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_settings(self, settings: AnimationSettings) -> None:
        self.duration_spin.setValue(settings.duration)
        self.loop_toggle.setChecked(settings.seamless_loop)
        self.fps_spin.setValue(settings.fps)
        self.width_spin.setValue(settings.width)
        self.height_spin.setValue(settings.height)
        self._set_combo(self.rotation_mode, settings.rotation_mode)
        self.spin_speed.setValue(settings.spin_speed)
        self._set_combo(self.spin_direction, settings.spin_direction)
        self.start_angle.setValue(settings.start_angle)
        self.elevation.setValue(settings.elevation)
        self.distance_scale.setValue(settings.distance_scale)
        # Convert legacy screen-relative pan values once when a preset is
        # loaded. All subsequent edits use a stable world-space orbit target.
        target = camera_target_offset(settings, 0.0)
        self.camera_target_x.setValue(float(target[0]))
        self.camera_target_y.setValue(float(target[1]))
        self.camera_target_z.setValue(float(target[2]))
        self._set_combo(self.rotation_center_mode, settings.rotation_center_mode)
        self.rotation_center_x.setValue(settings.rotation_center_x)
        self.rotation_center_y.setValue(settings.rotation_center_y)
        self.rotation_center_z.setValue(settings.rotation_center_z)
        self.fov.setValue(settings.fov)
        self._set_combo(self.start_representation, settings.start_representation)
        self._set_combo(
            self.trip_mode,
            settings.round_trip if settings.transformation_enabled else None,
        )
        self.transition_start_spin.setValue(settings.transition_start)
        self.transition_duration_spin.setValue(settings.transition_duration)
        self.return_hold_spin.setValue(settings.return_hold)
        self._set_combo(self.easing_combo, settings.easing)
        self._set_combo(self.transition_effect, settings.transition_effect)
        self._set_combo(self.scan_direction, settings.scan_direction)
        self._set_combo(self.up_axis, settings.up_axis)
        self.scan_feather_spin.setValue(settings.scan_feather)
        self.point_radius.setValue(settings.point_radius)
        self.point_opacity.setValue(settings.point_opacity)
        self.splat_scale.setValue(settings.splat_scale)
        self.splat_opacity.setValue(settings.splat_opacity)
        self._min_splat_pixels = settings.min_splat_pixels
        self._max_splat_pixels = settings.max_splat_pixels
        self.exposure.setValue(settings.exposure)
        self._background = settings.background
        self.gradient.setValue(settings.background_gradient)
        self._set_combo(self.codec_combo, settings.codec)
        self.quality_spin.setValue(settings.quality)
        self._update_background_button()
        self._match_resolution_preset()
        self._trip_mode_changed()
        self._rotation_center_mode_changed()
        self._settings_changed()

    def _settings_changed(self, *_args: object) -> None:
        duration = self._settings().duration
        if self._play_seconds > duration:
            self._play_seconds %= duration
        self._update_timeline_summary()
        self.preview_pending = True
        interactive = (
            self._preview_fast or self.play_timer.isActive() or self.timeline.isSliderDown()
        )
        self.preview_timer.start(1 if interactive else 60)

    def _update_timeline_summary(self) -> None:
        selected = self._settings(resolve_loop=False)
        settings = selected.resolved_for_loop()
        if not settings.transformation_enabled:
            representation = (
                "Gaussian splat" if settings.start_representation == "splat" else "Point cloud"
            )
            text = f"No transition. Showing {representation.lower()} for the full video."
        elif settings.round_trip:
            text = (
                f"Transition: {settings.transition_start:.1f} to "
                f"{settings.transition_start + settings.transition_duration:.1f} s. "
                f"Return: {settings.return_start:.1f} to {settings.transition_end:.1f} s."
            )
        else:
            text = (
                f"Transition: {settings.transition_start:.1f} to "
                f"{settings.transition_start + settings.transition_duration:.1f} s."
            )
        if selected.seamless_loop:
            turns = abs(settings.spin_speed) * settings.duration / 360.0
            loop_text = f"Loop: {turns:.0f} complete turn"
            if round(turns) != 1:
                loop_text += "s"
            loop_text += f", {settings.duration:.2f} s at {settings.spin_speed:.2f}°/s"
            if (
                abs(settings.duration - selected.duration) > 1e-6
                or abs(settings.spin_speed - selected.spin_speed) > 1e-6
            ):
                loop_text += (
                    f" (selected: {selected.duration:.2f} s at "
                    f"{selected.spin_speed:.2f}°/s)"
                )
            text += "\n" + loop_text

        if settings.transformation_enabled and settings.transition_end > settings.duration:
            text += " Transition extends past the end of the video."
            self.timeline_summary.setObjectName("error")
        else:
            self.timeline_summary.setObjectName("hint")
        self.timeline_summary.setText(text)
        self.timeline_summary.style().unpolish(self.timeline_summary)
        self.timeline_summary.style().polish(self.timeline_summary)
        self.render_estimate.setText(
            f"{settings.frame_count:,} frames, {settings.width:,} × "
            f"{settings.height:,}, {settings.fps} fps"
        )
        self._update_time_label()

    def _trip_mode_changed(self, *_args: object) -> None:
        mode = self.trip_mode.currentData()
        if self.loop_toggle.isChecked() and mode is False:
            self._set_combo(self.trip_mode, True)
            return
        transformation_enabled = mode is not None
        for widget in (
            self.transition_start_spin,
            self.transition_duration_spin,
            self.easing_combo,
            self.transition_effect,
            self.scan_direction,
            self.scan_feather_spin,
        ):
            widget.setEnabled(transformation_enabled)
        self.return_hold_spin.setEnabled(transformation_enabled and mode is True)
        self._settings_changed()

    def _rotation_center_mode_changed(self, *_args: object) -> None:
        custom_center = self.rotation_center_mode.currentData() == "custom"
        for widget in (
            self.rotation_center_x,
            self.rotation_center_y,
            self.rotation_center_z,
        ):
            widget.setEnabled(custom_center)
        self._settings_changed()

    def _loop_toggled(self, checked: bool) -> None:
        if checked and self.trip_mode.currentData() is False:
            self._set_combo(self.trip_mode, True)
        self._trip_mode_changed()

    def _preview_size(
        self,
        settings: AnimationSettings,
        *,
        interactive: bool = False,
        seconds: float | None = None,
    ) -> tuple[int, int]:
        representation = settings.representation_at(
            self._play_seconds if seconds is None else seconds
        )
        pure_point_frame = representation.source == 0.0 and representation.target == 0.0
        point_supersampling = (1.25 if interactive else 1.6) if pure_point_frame else 1.0
        pixel_ratio = max(1.0, float(self.preview.devicePixelRatioF())) * point_supersampling
        quality_scale = 0.55 if interactive else 1.0
        available_width = max(320, round((self.preview.width() - 8) * pixel_ratio * quality_scale))
        available_height = max(
            240, round((self.preview.height() - 8) * pixel_ratio * quality_scale)
        )
        aspect = settings.width / settings.height
        if pure_point_frame:
            width_cap, height_cap = (1100, 720) if interactive else (1920, 1200)
        else:
            width_cap, height_cap = (900, 640) if interactive else (1440, 1000)
        width = min(width_cap, available_width)
        height = round(width / aspect)
        if height > min(height_cap, available_height):
            height = min(height_cap, available_height)
            width = round(height * aspect)
        return max(16, width - width % 2), max(16, height - height % 2)

    def _request_preview(self) -> None:
        if not self.scene_loaded or self.render_active:
            return
        if self.preview_inflight:
            self.preview_pending = True
            return
        settings = self._settings()
        interactive = (
            self._preview_fast or self.play_timer.isActive() or self.timeline.isSliderDown()
        )
        width, height = self._preview_size(settings, interactive=interactive)
        self.preview_inflight = True
        self.preview_pending = False
        self._last_preview_target_size = (width, height)
        # Interaction performance comes from temporary resolution scaling,
        # never from stale draw order or dropping Gaussians.
        self.requestPreview.emit(settings, self._play_seconds, width, height, True)

    @Slot(QImage)
    def _preview_ready(self, image: QImage) -> None:
        self.preview_inflight = False
        if not self.scene_loaded:
            return
        self.preview.set_preview(image)
        if self._preview_fast or self.play_timer.isActive() or self.timeline.isSliderDown():
            self.status_label.setText("Updating preview; full quality resumes when you stop")
        else:
            self.status_label.setText("Preview ready. Use WASD and Q/E to move.")
        if self.preview_pending or self.play_timer.isActive():
            self.preview_timer.start(1)

    def _preview_resized(self) -> None:
        if not self.scene_loaded or self.render_active:
            return
        settings = self._settings()
        interactive = (
            self._preview_fast or self.play_timer.isActive() or self.timeline.isSliderDown()
        )
        target_size = self._preview_size(settings, interactive=interactive)
        if target_size == self._last_preview_target_size:
            return
        self.preview_pending = True
        self.preview_timer.start(120)

    def _choose_project(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose project folder", self.project_line.text()
        )
        if folder:
            self.open_project(Path(folder))

    def open_project(self, path: Path) -> None:
        self.project_line.setText(str(path))
        self._scan_project(True)

    def _scan_project(self, auto_load: bool = False) -> None:
        try:
            self.detection = detect_project(self.project_line.text())
        except Exception as exc:
            self.source_info.setObjectName("error")
            self.source_info.setText(str(exc))
            return
        self.candidate_combo.blockSignals(True)
        self.candidate_combo.clear()
        for candidate in self.detection.candidates:
            detail = candidate.info.description if candidate.info else "Unknown format"
            self.candidate_combo.addItem(f"{candidate.label} ({detail})", str(candidate.path))
        self.candidate_combo.blockSignals(False)
        if not self.detection.candidates:
            self.source_info.setObjectName("error")
            self.source_info.setText("No supported .ply or .splat file found in this project")
            return
        self.candidate_combo.setCurrentIndex(0)
        self._candidate_changed(0)
        project_name = self.detection.root.name.replace(" ", "-").lower()
        extension = expected_extension(str(self.codec_combo.currentData()))
        self.output_line.setText(
            str(self.detection.suggested_output / f"{project_name}-transition{extension}")
        )
        if auto_load:
            self._load_scene()

    def _candidate_changed(self, index: int) -> None:
        if index < 0:
            return
        path = self.candidate_combo.itemData(index)
        if not path:
            return
        self.source_line.setText(str(path))
        self._inspect_manual_source()

    def _choose_source(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Gaussian splat",
            self.source_line.text(),
            "Gaussian splats (*.ply *.splat);;PLY files (*.ply);;SPLAT files (*.splat)",
        )
        if path:
            self.open_source(Path(path), auto_load=True)

    def open_source(self, path: Path, *, auto_load: bool = True) -> None:
        self.source_line.setText(str(path))
        self._inspect_manual_source()
        if not self.output_line.text():
            self.output_line.setText(str(path.with_name(path.stem + "-transition.mp4")))
        if auto_load:
            self._load_scene()

    def _inspect_manual_source(self) -> None:
        path = Path(self.source_line.text()).expanduser()
        try:
            info = inspect_source(path)
        except Exception as exc:
            self.source_info.setObjectName("error")
            self.source_info.setText(str(exc))
        else:
            self.source_info.setObjectName("success")
            self.source_info.setText(info.description)
        self.source_info.style().unpolish(self.source_info)
        self.source_info.style().polish(self.source_info)

    def _load_scene(self) -> None:
        if self.render_active:
            return
        source = Path(self.source_line.text()).expanduser()
        if not source.is_file():
            QMessageBox.warning(
                self, "Source missing", "Choose an existing .ply or .splat source first."
            )
            return
        self.scene_loaded = False
        self.preview_inflight = False
        self.preview_pending = False
        self._preview_fast = False
        self._last_preview_target_size = None
        self.preview_settle_timer.stop()
        self.render_button.setEnabled(False)
        self.load_scene_button.setEnabled(False)
        self.preview.clear_preview("Loading scene...")
        self.status_label.setText(f"Loading {source.name}...")
        self.requestLoad.emit(str(source), self.budget_spin.value())

    def _set_gpu_reading(
        self,
        name: str,
        fraction: float | None,
        value_text: str,
    ) -> None:
        self.gpu_name_label.setText(_short_gpu_name(name))
        self.gpu_name_label.setToolTip(name)
        self.gpu_usage_label.setText(value_text)
        if fraction is None:
            self.gpu_meter.setValue(0)
            color = COLORS["faint"]
        else:
            fraction = max(0.0, min(1.0, fraction))
            self.gpu_meter.setValue(round(fraction * 1000))
            color = (
                COLORS["green"]
                if fraction < 0.7
                else "#ffbd52"
                if fraction < 0.9
                else COLORS["red"]
            )
        self.gpu_meter.setStyleSheet(
            "QProgressBar#gpuMeter { border:0; border-radius:2px; "
            f"background:{COLORS['line_soft']}; }} "
            "QProgressBar#gpuMeter::chunk { border-radius:2px; "
            f"background:{color}; }}"
        )

    def _request_gpu_update(self) -> None:
        if not self._nvidia_smi:
            self._set_gpu_reading(
                self._renderer_name or "GPU",
                None,
                "usage unavailable" if self._renderer_name else "not loaded",
            )
            return
        if self.gpu_process.state() != QProcess.ProcessState.NotRunning:
            return
        self.gpu_process.start(
            self._nvidia_smi,
            [
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
        )

    def _gpu_query_finished(self, exit_code: int, *_args: object) -> None:
        output = bytes(self.gpu_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        line = next((item.strip() for item in output.splitlines() if item.strip()), "")
        fields = [item.strip() for item in line.rsplit(",", 2)]
        if exit_code != 0 or len(fields) != 3:
            self._set_gpu_reading(
                self._renderer_name or "GPU",
                None,
                "usage unavailable" if self._renderer_name else "not loaded",
            )
            return
        try:
            used_mib, total_mib = float(fields[1]), float(fields[2])
        except ValueError:
            self._set_gpu_reading(fields[0] or self._renderer_name or "GPU", None, "unknown")
            return
        fraction = used_mib / total_mib if total_mib > 0 else 0.0
        self._set_gpu_reading(
            fields[0],
            fraction,
            f"{used_mib / 1024:.1f} / {total_mib / 1024:.0f} GB",
        )

    @Slot(object)
    def _scene_loaded(self, details: dict[str, object]) -> None:
        self.scene_loaded = True
        self.load_scene_button.setEnabled(True)
        self.render_button.setEnabled(True)
        loaded = int(details["loaded"])
        original = int(details["original"])
        subset = f"{loaded:,} / {original:,}" if loaded != original else f"{loaded:,}"
        self.loaded_label.setText(f"{subset} points")
        self._renderer_name = str(details["renderer"])
        self._request_gpu_update()
        self.source_info.setObjectName("success")
        self.source_info.setText(str(details["description"]))
        self.status_label.setText("File loaded. Building preview...")
        self.preview_timer.start(1)

    def _worker_failed(self, message: str) -> None:
        self.preview_inflight = False
        self.load_scene_button.setEnabled(True)
        if self.scene_loaded:
            self.render_button.setEnabled(True)
        if self.render_active:
            self.render_active = False
            self.cancel_button.setEnabled(False)
            self.render_button.setEnabled(self.scene_loaded)
            self.render_status.setText(message)
        self.status_label.setText(message)
        if "cancelled" not in message.lower():
            QMessageBox.critical(self, APP_NAME, message)

    def _timeline_changed(self, value: int) -> None:
        if self.timeline.isSliderDown():
            self._preview_interaction_started()
        self._play_seconds = value / 10_000 * self._settings().duration
        self._update_time_label()
        self.preview_timer.start(40)

    def _update_time_label(self) -> None:
        duration = self._settings().duration

        def stamp(seconds: float) -> str:
            minutes = int(seconds // 60)
            return f"{minutes}:{seconds - minutes * 60:04.1f}"

        self.time_label.setText(f"{stamp(self._play_seconds)} / {stamp(duration)}")

    def _toggle_play(self) -> None:
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_button.setText("▶")
            self._preview_interaction_settled()
        else:
            self._last_tick = time.monotonic()
            self._preview_fast = True
            self.preview_settle_timer.stop()
            self.play_timer.start()
            self.play_button.setText("Ⅱ")

    def _play_tick(self) -> None:
        now = time.monotonic()
        self._play_seconds += now - self._last_tick
        self._last_tick = now
        duration = self._settings().duration
        if self._play_seconds >= duration:
            self._play_seconds %= duration
        self.timeline.blockSignals(True)
        self.timeline.setValue(round(self._play_seconds / duration * 10_000))
        self.timeline.blockSignals(False)
        self._update_time_label()
        self.preview_timer.start(1)

    def _preview_dragged(self, x: float, y: float) -> None:
        self._preview_interaction_started()
        angle = self.start_angle.value() + x * 0.35
        while angle > 360:
            angle -= 720
        while angle < -360:
            angle += 720
        self.start_angle.setValue(angle)
        self.elevation.setValue(max(-80.0, min(80.0, self.elevation.value() + y * 0.22)))

    def _preview_zoomed(self, delta: float) -> None:
        self._preview_interaction_started()
        factor = 0.9 ** (delta / 120.0)
        self.distance_scale.setValue(self.distance_scale.value() * factor)

    def _preview_panned(self, x: float, y: float) -> None:
        self._preview_interaction_started()
        settings = self._settings()
        camera_right, camera_up, _forward = camera_basis(settings, self._play_seconds)
        pixmap = self.preview.pixmap()
        display_height = (
            pixmap.height() if pixmap is not None and not pixmap.isNull() else self.preview.height()
        )
        units_per_pixel = (
            2.0
            * settings.distance_scale
            * np.tan(np.radians(settings.fov) * 0.5)
            / max(1, display_height)
        )
        target = self._camera_target()
        target += -x * units_per_pixel * camera_right + y * units_per_pixel * camera_up
        self._set_camera_target(target)

    def _preview_navigated(
        self,
        right: float,
        upward: float,
        forward: float,
        elapsed: float,
    ) -> None:
        self._preview_interaction_started()
        settings = self._settings()
        camera_right, camera_up, camera_forward = camera_basis(settings, self._play_seconds)
        move = right * camera_right + upward * camera_up + forward * camera_forward
        norm = float(np.linalg.norm(move))
        if norm <= 1e-12:
            return
        input_speed = max(abs(right), abs(upward), abs(forward))
        local_scale = float(np.clip(settings.distance_scale, 0.001, 1.0))
        target = self._camera_target()
        target += move / norm * local_scale * 0.45 * input_speed * max(0.0, elapsed)
        self._set_camera_target(target)

    def _camera_target(self) -> np.ndarray:
        return np.array(
            [
                self.camera_target_x.value(),
                self.camera_target_y.value(),
                self.camera_target_z.value(),
            ],
            dtype=np.float64,
        )

    def _set_camera_target(self, target: np.ndarray) -> None:
        widgets = (self.camera_target_x, self.camera_target_y, self.camera_target_z)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            for widget, value in zip(widgets, target, strict=True):
                widget.setValue(float(value))
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        self._settings_changed()

    def _preview_interaction_started(self) -> None:
        self._preview_fast = True
        self.preview_settle_timer.start()

    def _preview_interaction_settled(self) -> None:
        self._preview_fast = False
        self.preview_pending = True
        self.preview_timer.start(1)

    def _show_opening_frame(self) -> None:
        self.play_timer.stop()
        self.play_button.setText("▶")
        self._play_seconds = 0.0
        self._preview_fast = False
        self.preview_settle_timer.stop()
        self.timeline.setValue(0)
        self._update_time_label()
        self.preview_timer.start(1)
        self.preview.setFocus(Qt.OtherFocusReason)

    def _reset_camera(self) -> None:
        defaults = AnimationSettings()
        self.start_angle.setValue(defaults.start_angle)
        self.elevation.setValue(defaults.elevation)
        self.distance_scale.setValue(defaults.distance_scale)
        self._set_camera_target(np.zeros(3, dtype=np.float64))
        self._set_combo(self.rotation_center_mode, defaults.rotation_center_mode)
        self.rotation_center_x.setValue(defaults.rotation_center_x)
        self.rotation_center_y.setValue(defaults.rotation_center_y)
        self.rotation_center_z.setValue(defaults.rotation_center_z)
        self.fov.setValue(defaults.fov)
        self._show_opening_frame()

    def _choose_background(self) -> None:
        color = QColorDialog.getColor(QColor(self._background), self, "Choose video background")
        if color.isValid():
            self._background = color.name()
            self._update_background_button()
            self._settings_changed()

    def _update_background_button(self) -> None:
        self.background_button.setText(self._background.upper())
        self.background_button.setStyleSheet(
            f"text-align: left; padding-left: 34px; background-color: {self._background};"
        )

    def _resolution_preset_changed(self, index: int) -> None:
        value = self.resolution_preset.itemData(index)
        if value:
            self.width_spin.setValue(value[0])
            self.height_spin.setValue(value[1])

    def _match_resolution_preset(self) -> None:
        size = (self.width_spin.value(), self.height_spin.value())
        for index in range(self.resolution_preset.count()):
            if self.resolution_preset.itemData(index) == size:
                self.resolution_preset.setCurrentIndex(index)
                return
        self.resolution_preset.setCurrentIndex(self.resolution_preset.count() - 1)

    def _codec_changed(self, *_args: object) -> None:
        path = Path(self.output_line.text()) if self.output_line.text() else None
        codec = str(self.codec_combo.currentData())
        if path:
            self.output_line.setText(str(path.with_suffix(expected_extension(codec))))
        self.quality_spin.setMaximum(63 if codec == "vp9" else 51)
        self.quality_spin.setEnabled(codec != "prores")
        self._settings_changed()

    def _choose_output(self) -> None:
        codec = str(self.codec_combo.currentData())
        suffix = expected_extension(codec)
        filters = {
            ".mp4": "MP4 video (*.mp4)",
            ".mov": "QuickTime video (*.mov)",
            ".webm": "WebM video (*.webm)",
        }
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save animation",
            self.output_line.text(),
            filters[suffix],
        )
        if path:
            self.output_line.setText(str(Path(path).with_suffix(suffix)))

    def _start_render(self) -> None:
        if not self.scene_loaded:
            return
        settings = self._settings()
        try:
            settings.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        output_text = self.output_line.text().strip()
        if not output_text:
            self._choose_output()
            output_text = self.output_line.text().strip()
            if not output_text:
                return
        output = Path(output_text).expanduser().with_suffix(expected_extension(settings.codec))
        if output.exists():
            answer = QMessageBox.question(
                self,
                "Replace video?",
                f"{output.name} already exists. Replace it?",
            )
            if answer != QMessageBox.Yes:
                return
        self.output_line.setText(str(output))
        self.render_active = True
        self.play_timer.stop()
        self.play_button.setText("▶")
        self.render_button.setEnabled(False)
        self.load_scene_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.render_progress.setValue(0)
        self.render_status.setText("Starting renderer...")
        self.status_label.setText("Rendering video...")
        self.requestRender.emit(settings, str(output))

    def _cancel_render(self) -> None:
        if self.render_active:
            self.cancel_button.setEnabled(False)
            self.render_status.setText("Cancelling after the current frame...")
            self.worker.cancel()

    @Slot(int, str)
    def _render_progress(self, value: int, message: str) -> None:
        self.render_progress.setValue(value)
        self.render_status.setText(message)

    @Slot(str)
    def _render_finished(self, path: str) -> None:
        self.render_active = False
        self.render_progress.setValue(100)
        self.render_status.setText(f"Saved {path}")
        self.status_label.setText("Video render complete")
        self.cancel_button.setEnabled(False)
        self.render_button.setEnabled(True)
        self.load_scene_button.setEnabled(True)
        QMessageBox.information(self, "Video complete", f"Animation saved to:\n{path}")

    def _save_preset(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save animation preset",
            "splat-animation.json",
            "JSON preset (*.json)",
        )
        if not path:
            return
        try:
            self._settings(resolve_loop=False).to_json(Path(path).with_suffix(".json"))
        except Exception as exc:
            QMessageBox.critical(self, "Could not save preset", str(exc))

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load animation preset",
            "",
            "JSON preset (*.json)",
        )
        if not path:
            return
        try:
            self._apply_settings(AnimationSettings.from_json(path))
        except Exception as exc:
            QMessageBox.critical(self, "Could not load preset", str(exc))

    def closeEvent(self, event: QCloseEvent) -> None:
        self.preview_timer.stop()
        self.preview_settle_timer.stop()
        self.play_timer.stop()
        self.gpu_timer.stop()
        if self.gpu_process.state() != QProcess.ProcessState.NotRunning:
            self.gpu_process.kill()
            self.gpu_process.waitForFinished(250)
        self.worker.cancel()
        if self.worker_thread.isRunning():
            self.requestShutdown.emit()
            self.worker_thread.wait()
        event.accept()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--project", type=Path, help="Open and auto-detect a project")
    parser.add_argument("--input", type=Path, help="Open a .ply or .splat file")
    parser.add_argument("--preset", type=Path, help="Load animation settings JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    application = QApplication(sys.argv[:1])
    application.setApplicationName(APP_NAME)
    application.setOrganizationName("Reconstrura")
    application.setWindowIcon(_app_icon())
    apply_theme(application)
    window = MainWindow()
    if arguments.preset:
        try:
            window._apply_settings(AnimationSettings.from_json(arguments.preset))
        except Exception as exc:
            message = str(exc)
            QTimer.singleShot(
                0,
                lambda: QMessageBox.critical(window, "Preset error", message),
            )
    if arguments.project:
        QTimer.singleShot(0, lambda: window.open_project(arguments.project))
    elif arguments.input:
        QTimer.singleShot(0, lambda: window.open_source(arguments.input, auto_load=True))
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
