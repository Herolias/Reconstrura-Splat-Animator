from __future__ import annotations

import threading
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage

from .io import load_scene
from .models import AnimationSettings
from .renderer import GpuRenderer
from .video import RenderCancelled, render_video


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
            data = self.renderer.render_frame(
                preview_settings,
                seconds,
                sort_depth=sort_depth,
            )
            components = 4 if preview_settings.transparent_background else 3
            array = (
                np.frombuffer(data, dtype=np.uint8).reshape(height, width, components)[::-1].copy()
            )
            image_format = (
                (
                    QImage.Format_RGBA8888_Premultiplied
                    if preview_settings.premultiplied_alpha
                    else QImage.Format_RGBA8888
                )
                if preview_settings.transparent_background
                else QImage.Format_RGB888
            )
            image = QImage(
                array.data,
                width,
                height,
                width * components,
                image_format,
            ).copy()
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
