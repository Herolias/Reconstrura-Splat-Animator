from __future__ import annotations

import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from splat_animator.io import load_scene
from splat_animator.models import AnimationSettings
from splat_animator.renderer import GpuRenderer, render_video, rotation_center_offset

from .helpers import write_gaussian_ply


def _renderer_or_skip(scene):
    try:
        return GpuRenderer(scene)
    except RuntimeError as exc:
        pytest.skip(f"No OpenGL 3.3 context available: {exc}")


def test_rotation_center_modes() -> None:
    orbit = replace(
        AnimationSettings(),
        camera_target_x=0.25,
        camera_target_y=-0.5,
        camera_target_z=0.75,
    )
    target = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    assert np.array_equal(rotation_center_offset(orbit, 0.0), target)
    assert np.array_equal(
        rotation_center_offset(replace(orbit, rotation_center_mode="target"), 0.0),
        target,
    )
    assert np.array_equal(
        rotation_center_offset(replace(orbit, rotation_center_mode="scene"), 0.0),
        np.zeros(3, dtype=np.float32),
    )
    assert np.array_equal(
        rotation_center_offset(replace(orbit, rotation_mode="object_spin"), 0.0),
        np.zeros(3, dtype=np.float32),
    )
    custom = replace(
        orbit,
        rotation_center_mode="custom",
        rotation_center_x=-1.0,
        rotation_center_y=2.0,
        rotation_center_z=3.0,
    )
    assert np.array_equal(
        rotation_center_offset(custom, 0.0),
        np.array([-1.0, 2.0, 3.0], dtype=np.float32),
    )


def test_gpu_frame_contains_rendered_scene(tmp_path: Path) -> None:
    scene = load_scene(write_gaussian_ply(tmp_path / "scene.ply", 300), budget=None)
    settings = replace(
        AnimationSettings(),
        width=128,
        height=96,
        duration=0.1,
        fps=10,
        distance_scale=2.0,
    )
    renderer = _renderer_or_skip(scene)
    try:
        frame = renderer.render_rgb(settings, 0.0)
        exact_depth_row = renderer._last_depth_row.copy()
        target_frame = renderer.render_rgb(replace(settings, camera_target_x=0.5), 0.0)
        # Translating the camera and orbit target cannot change relative depth.
        assert np.array_equal(renderer._last_depth_row, exact_depth_row)
        renderer.render_rgb(
            replace(settings, start_angle=2.0),
            0.0,
            sort_depth=False,
        )
        assert np.array_equal(renderer._last_depth_row, exact_depth_row)
        renderer.render_rgb(replace(settings, start_angle=2.0), 0.0)
        assert not np.array_equal(renderer._last_depth_row, exact_depth_row)
        panned_frame = renderer.render_rgb(replace(settings, camera_pan_x=0.5), 0.0)
        close_up_frame = renderer.render_rgb(replace(settings, distance_scale=0.2), 0.0)
        scene_center_spin = renderer.render_rgb(
            replace(settings, rotation_mode="object_spin", start_angle=35.0),
            0.0,
        )
        custom_center_spin = renderer.render_rgb(
            replace(
                settings,
                rotation_mode="object_spin",
                start_angle=35.0,
                rotation_center_mode="custom",
                rotation_center_x=0.5,
            ),
            0.0,
        )
        effect_frames = {
            effect: renderer.render_rgb(
                replace(
                    settings,
                    transition_effect=effect,
                    transition_start=0.0,
                    transition_duration=2.0,
                ),
                1.0,
            )
            for effect in ("sweep", "radial", "wave", "spiral", "dissolve")
        }
        pixel_scale = renderer.program["u_pixel_scale"].value
        point_radius = renderer.program["u_point_radius"].value
    finally:
        renderer.close()
    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(96, 128, 3)
    assert len(frame) == 128 * 96 * 3
    assert float(pixels.std()) > 1.0
    assert target_frame != frame
    assert panned_frame != frame
    assert close_up_frame != frame
    assert custom_center_spin != scene_center_spin
    assert all(len(effect_frame) == len(frame) for effect_frame in effect_frames.values())
    assert len(set(effect_frames.values())) == len(effect_frames)
    assert pixel_scale == pytest.approx(96 / 1080)
    assert point_radius == pytest.approx(settings.point_radius * 96 / 1080)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_ffmpeg_video_export(tmp_path: Path) -> None:
    scene = load_scene(write_gaussian_ply(tmp_path / "scene.ply", 100), budget=None)
    settings = replace(
        AnimationSettings(),
        width=128,
        height=96,
        duration=0.2,
        fps=10,
        transition_start=0.0,
        transition_duration=0.2,
        round_trip=False,
    )
    output = tmp_path / "animation.mp4"
    renderer = _renderer_or_skip(scene)
    try:
        rendered = render_video(renderer, settings, output)
    finally:
        renderer.close()
    assert rendered == output
    assert output.stat().st_size > 500
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream=color_range,color_space,color_transfer,color_primaries",
                "-of",
                "default=noprint_wrappers=1",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "color_range=tv" in probe.stdout
        assert "color_space=bt709" in probe.stdout
        assert "color_transfer=bt709" in probe.stdout
        assert "color_primaries=bt709" in probe.stdout
