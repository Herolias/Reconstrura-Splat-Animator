from __future__ import annotations

import os
import time
from dataclasses import replace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, Qt  # noqa: E402
from PySide6.QtGui import QImage, QKeyEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from splat_animator.app import MainWindow, _short_gpu_name  # noqa: E402
from splat_animator.models import AnimationSettings  # noqa: E402
from splat_animator.theme import apply_theme  # noqa: E402


def test_main_window_starts_and_stops_render_thread() -> None:
    application = QApplication.instance() or QApplication([])
    apply_theme(application)
    window = MainWindow()
    window.show()
    application.processEvents()
    assert window.windowTitle() == "Reconstrura Splat Animator"
    assert not window.windowIcon().isNull()
    assert window.worker_thread.isRunning()
    assert not window.premultiplied_alpha.isChecked()
    assert not window.premultiplied_alpha.isEnabled()
    window.transparent_background.setChecked(True)
    application.processEvents()
    assert window.premultiplied_alpha.isChecked()
    assert window.premultiplied_alpha.isEnabled()
    window.transparent_background.setChecked(False)
    application.processEvents()
    assert not window.premultiplied_alpha.isChecked()
    assert not window.premultiplied_alpha.isEnabled()
    assert window.inspector.horizontalScrollBar().maximum() == 0
    assert window.inspector.widget().width() == window.inspector.viewport().width()
    preview_size = window.preview.size()
    preview_hint = window.preview.sizeHint()
    preview_resizes: list[tuple[int, int]] = []
    window.preview.resized.connect(
        lambda: preview_resizes.append((window.preview.width(), window.preview.height()))
    )
    window.preview.set_preview(QImage(900, 600, QImage.Format_RGB888))
    application.processEvents()
    window.preview.set_preview(QImage(495, 330, QImage.Format_RGB888))
    application.processEvents()
    for message in (
        "Updating preview...",
        "Preview ready. Use WASD and Q/E to move.",
    ) * 3:
        window.status_label.setText(message)
        application.processEvents()
    assert window.preview.size() == preview_size
    assert window.preview.sizeHint() == preview_hint
    assert not preview_resizes

    window.scene_loaded = True
    window._last_preview_target_size = window._preview_size(window._settings())
    window.preview_pending = False
    window._preview_resized()
    assert not window.preview_pending
    window.scene_loaded = False
    assert window.trip_mode.currentData() is True
    assert window.budget_spin.value() == 0
    assert window.distance_scale.minimum() == 0.0001
    assert window.fov.minimum() == 1.0
    assert window.rotation_center_mode.currentData() == "automatic"
    assert not window.rotation_center_x.isEnabled()
    preset_values = [
        window.resolution_preset.itemData(index)
        for index in range(window.resolution_preset.count())
    ]
    assert (2160, 3840) in preset_values
    visible_labels = [label.text() for label in window.findChildren(type(window.gpu_name_label))]
    assert all("—" not in label for label in visible_labels)
    assert "OPEN SOURCE · MIT" not in visible_labels
    window.loop_toggle.setChecked(True)
    application.processEvents()
    loop_settings = window._settings()
    assert window.trip_mode.isEnabled()
    assert loop_settings.round_trip
    assert loop_settings.spin_speed * loop_settings.duration / 360.0 == 1.0

    window._set_combo(window.trip_mode, None)
    window._set_combo(window.start_representation, "point")
    application.processEvents()
    static_settings = window._settings()
    assert not static_settings.transformation_enabled
    assert static_settings.representation_at(100.0).source == 0.0
    assert not window.transition_effect.isEnabled()
    assert not window.transition_duration_spin.isEnabled()
    window._set_combo(window.trip_mode, True)

    splat_preview_size = window._preview_size(
        replace(
            window._settings(resolve_loop=False),
            transformation_enabled=False,
            start_representation="splat",
        )
    )
    point_preview_size = window._preview_size(
        replace(
            window._settings(resolve_loop=False),
            transformation_enabled=False,
            start_representation="point",
        )
    )
    assert point_preview_size[0] > splat_preview_size[0]
    assert point_preview_size[1] > splat_preview_size[1]

    window._set_combo(window.rotation_center_mode, "custom")
    window.rotation_center_x.setValue(0.25)
    window.rotation_center_y.setValue(-0.5)
    window.rotation_center_z.setValue(0.75)
    application.processEvents()
    assert window.rotation_center_x.isEnabled()
    centered = window._settings(resolve_loop=False)
    assert centered.rotation_center_mode == "custom"
    assert centered.rotation_center_x == 0.25
    assert centered.rotation_center_y == -0.5
    assert centered.rotation_center_z == 0.75
    window._set_combo(window.rotation_center_mode, "target")
    application.processEvents()
    assert not window.rotation_center_x.isEnabled()

    starting_distance = window.distance_scale.value()
    window._preview_navigated(1.0, 1.0, 1.0, 0.25)
    assert window._preview_fast
    assert window.preview_settle_timer.isActive()
    target = (
        window.camera_target_x.value(),
        window.camera_target_y.value(),
        window.camera_target_z.value(),
    )
    assert any(abs(value) > 0.0 for value in target)
    assert window.distance_scale.value() == starting_distance
    moved = window._settings(resolve_loop=False)
    assert moved.camera_target_x == window.camera_target_x.value()
    assert moved.camera_target_y == window.camera_target_y.value()
    assert moved.camera_target_z == window.camera_target_z.value()
    window._preview_interaction_settled()
    assert not window._preview_fast

    movements: list[tuple[float, float, float, float]] = []
    window.preview.navigated.connect(lambda *movement: movements.append(movement))
    press = QKeyEvent(QEvent.KeyPress, Qt.Key_W, Qt.NoModifier)
    window.preview.keyPressEvent(press)
    window.preview._last_navigation_tick = time.monotonic() - 0.04
    window.preview._emit_navigation()
    release = QKeyEvent(QEvent.KeyRelease, Qt.Key_W, Qt.NoModifier)
    window.preview.keyReleaseEvent(release)
    assert movements
    assert movements[-1][2] == 1.0

    window._reset_camera()
    assert window.camera_target_x.value() == 0.0
    assert window.camera_target_y.value() == 0.0
    assert window.camera_target_z.value() == 0.0
    assert window.distance_scale.value() == 2.65
    assert window.rotation_center_mode.currentData() == "automatic"
    assert window.rotation_center_x.value() == 0.0
    assert window.rotation_center_y.value() == 0.0
    assert window.rotation_center_z.value() == 0.0

    preset = replace(
        AnimationSettings(),
        codec="vp9",
        quality=63,
        transparent_background=True,
        premultiplied_alpha=True,
        min_splat_pixels=1.25,
        max_splat_pixels=48.0,
    )
    window._apply_settings(preset)
    restored = window._settings(resolve_loop=False)
    assert window.quality_spin.maximum() == 63
    assert restored.quality == 63
    assert restored.transparent_background
    assert restored.premultiplied_alpha
    assert window.premultiplied_alpha.isEnabled()
    assert not window.background_button.isEnabled()
    assert not window.gradient.isEnabled()
    assert restored.min_splat_pixels == 1.25
    assert restored.max_splat_pixels == 48.0
    window._set_combo(window.codec_combo, "h264")
    application.processEvents()
    assert not window.transparent_background.isChecked()
    assert not window.premultiplied_alpha.isChecked()
    assert not window.premultiplied_alpha.isEnabled()
    window.close()
    assert not window.worker_thread.isRunning()


def test_gpu_name_matches_reconstrura_compact_scheme() -> None:
    assert _short_gpu_name("NVIDIA GeForce RTX 4090") == "RTX 4090"
    assert _short_gpu_name("NVIDIA GeForce RTX 3070 Laptop GPU/PCIe/SSE2") == (
        "RTX 3070 Laptop GPU"
    )
