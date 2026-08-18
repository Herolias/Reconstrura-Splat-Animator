from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QProcess,
    Qt,
    QThread,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
)
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QMainWindow,
    QMessageBox,
)

from .camera import camera_basis, camera_target_offset
from .gui_assets import APP_NAME
from .gui_assets import app_icon as _app_icon
from .gui_assets import short_gpu_name as _short_gpu_name
from .io import ProjectDetection, detect_project, inspect_source
from .models import AnimationSettings
from .preview import PreviewCanvas
from .theme import COLORS, apply_theme
from .ui import MainWindowUiMixin
from .video import expected_extension
from .worker import RenderWorker

__all__ = [
    "APP_NAME",
    "MainWindow",
    "PreviewCanvas",
    "RenderWorker",
    "_app_icon",
    "_short_gpu_name",
    "main",
]


class MainWindow(QMainWindow, MainWindowUiMixin):
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
            transparent_background=self.transparent_background.isChecked(),
            premultiplied_alpha=self.premultiplied_alpha.isChecked(),
            codec=str(self.codec_combo.currentData()),
            quality=self.quality_spin.value(),
            bitrate_mbps=self.bitrate_spin.value(),
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
        self.transparent_background.setChecked(settings.transparent_background)
        self.premultiplied_alpha.setChecked(settings.premultiplied_alpha)
        self.quality_spin.setValue(settings.quality)
        self.bitrate_spin.setValue(settings.bitrate_mbps)
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
                    f" (selected: {selected.duration:.2f} s at {selected.spin_speed:.2f}°/s)"
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
        estimate = (
            f"{settings.frame_count:,} frames, {settings.width:,} × "
            f"{settings.height:,}, {settings.fps} fps"
        )
        if settings.bitrate_mbps > 0:
            estimated_size = settings.bitrate_mbps * settings.duration / 8.0
            estimate += f", about {estimated_size:,.1f} MB"
        self.render_estimate.setText(estimate)
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

    def _transparency_changed(self, checked: bool) -> None:
        if checked and self.codec_combo.currentData() != "vp9":
            self._set_combo(self.codec_combo, "vp9")
        self.premultiplied_alpha.setChecked(checked)
        self.premultiplied_alpha.setEnabled(checked)
        self.background_button.setEnabled(not checked)
        self.gradient.setEnabled(not checked)
        self._settings_changed()

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
        if self.transparent_background.isChecked() and codec != "vp9":
            self.transparent_background.setChecked(False)
        if path:
            self.output_line.setText(str(path.with_suffix(expected_extension(codec))))
        self.quality_spin.setMaximum(63 if codec == "vp9" else 51)
        if codec == "prores" and self.bitrate_spin.value() > 0:
            self.bitrate_spin.setValue(0.0)
        self.bitrate_spin.setEnabled(codec != "prores")
        self.quality_spin.setEnabled(codec != "prores" and self.bitrate_spin.value() == 0)
        self._settings_changed()

    def _bitrate_changed(self, bitrate: float) -> None:
        self.quality_spin.setEnabled(self.codec_combo.currentData() != "prores" and bitrate == 0)
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
