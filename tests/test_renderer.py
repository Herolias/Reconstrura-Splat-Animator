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


def test_degree_three_sh_color_changes_with_view_direction(tmp_path: Path) -> None:
    scene = load_scene(
        write_gaussian_ply(tmp_path / "degree-three.ply", count=1, sh_degree=3),
        budget=None,
    )
    settings = replace(
        AnimationSettings(),
        width=64,
        height=64,
        transformation_enabled=False,
        start_angle=0.0,
        distance_scale=2.0,
    )
    renderer = _renderer_or_skip(scene)
    try:
        front = renderer.render_rgb(settings, 0.0)
        side = renderer.render_rgb(replace(settings, start_angle=90.0), 0.0)
        assert renderer.sh_degree == 3
        assert renderer.sh_coefficient_count == 15
    finally:
        renderer.close()

    assert front != side


def test_transparent_frame_contains_straight_alpha(tmp_path: Path) -> None:
    scene = load_scene(write_gaussian_ply(tmp_path / "scene.ply", 100), budget=None)
    settings = replace(
        AnimationSettings(),
        width=64,
        height=64,
        transformation_enabled=False,
        transparent_background=True,
        premultiplied_alpha=False,
        codec="vp9",
    )
    renderer = _renderer_or_skip(scene)
    try:
        frame = renderer.render_frame(settings, 0.0)
    finally:
        renderer.close()

    pixels = np.frombuffer(frame, dtype=np.uint8).reshape(64, 64, 4)
    assert len(frame) == 64 * 64 * 4
    assert np.all(pixels[0, 0] == 0)
    assert int(pixels[:, :, 3].max()) > 0
    assert np.any((pixels[:, :, 3] > 0) & (pixels[:, :, 3] < 255))


def test_transparent_frame_matches_opaque_frame_when_composited(tmp_path: Path) -> None:
    scene = load_scene(write_gaussian_ply(tmp_path / "composite-scene.ply", 100), budget=None)
    opaque_settings = replace(
        AnimationSettings(),
        width=64,
        height=64,
        transformation_enabled=False,
        background="#000000",
        background_gradient=0.0,
    )
    transparent_settings = replace(
        opaque_settings,
        transparent_background=True,
        premultiplied_alpha=False,
        codec="vp9",
    )
    renderer = _renderer_or_skip(scene)
    try:
        opaque = renderer.render_rgb(opaque_settings, 0.0)
        transparent = renderer.render_frame(transparent_settings, 0.0)
    finally:
        renderer.close()

    opaque_pixels = np.frombuffer(opaque, dtype=np.uint8).reshape(64, 64, 3)
    rgba = np.frombuffer(transparent, dtype=np.uint8).reshape(64, 64, 4)
    composited = np.rint(
        rgba[:, :, :3].astype(np.float32) * rgba[:, :, 3:4].astype(np.float32) / 255.0
    ).astype(np.uint8)
    difference = np.abs(composited.astype(np.int16) - opaque_pixels.astype(np.int16))
    assert int(difference.max()) <= 2


def test_premultiplied_alpha_rgb_matches_black_background(tmp_path: Path) -> None:
    scene = load_scene(write_gaussian_ply(tmp_path / "premultiplied-scene.ply", 100), budget=None)
    opaque_settings = replace(
        AnimationSettings(),
        width=64,
        height=64,
        transformation_enabled=False,
        background="#000000",
        background_gradient=0.0,
    )
    transparent_settings = replace(
        opaque_settings,
        transparent_background=True,
        premultiplied_alpha=True,
        codec="vp9",
    )
    renderer = _renderer_or_skip(scene)
    try:
        opaque = renderer.render_rgb(opaque_settings, 0.0)
        transparent = renderer.render_frame(transparent_settings, 0.0)
    finally:
        renderer.close()

    opaque_pixels = np.frombuffer(opaque, dtype=np.uint8).reshape(64, 64, 3)
    rgba = np.frombuffer(transparent, dtype=np.uint8).reshape(64, 64, 4)
    difference = np.abs(rgba[:, :, :3].astype(np.int16) - opaque_pixels.astype(np.int16))
    assert int(difference.max()) <= 1
    assert int(rgba[:, :, 3].min()) == 0
    assert int(rgba[:, :, 3].max()) > 0


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


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_vp9_export_marks_alpha_channel(tmp_path: Path) -> None:
    scene = load_scene(write_gaussian_ply(tmp_path / "alpha-vp9-scene.ply", 50), budget=None)
    settings = replace(
        AnimationSettings(),
        width=64,
        height=64,
        duration=0.1,
        fps=10,
        transformation_enabled=False,
        transparent_background=True,
        codec="vp9",
    )
    output = tmp_path / "alpha.webm"
    renderer = _renderer_or_skip(scene)
    try:
        source_frame = renderer.render_frame(settings, 0.0)
        rendered = render_video(renderer, settings, output)
    finally:
        renderer.close()

    assert rendered == output
    assert output.stat().st_size > 500
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    decoded = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-c:v",
            "libvpx-vp9",
            "-i",
            str(output),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    decoded_pixels = np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(64, 64, 4)
    decoded_alpha = decoded_pixels[:, :, 3]
    assert int(decoded_alpha.min()) == 0
    assert int(decoded_alpha.max()) > 0

    # Video export flips OpenGL's bottom-up frame before encoding. Comparing
    # premultiplied pixels verifies both color and alpha while avoiding large,
    # meaningless straight-RGB differences in nearly transparent pixels.
    source_pixels = np.frombuffer(source_frame, dtype=np.uint8).reshape(64, 64, 4)[::-1]
    source_composite = (
        source_pixels[:, :, :3].astype(np.float32)
        * source_pixels[:, :, 3:4].astype(np.float32)
        / 255.0
    )
    decoded_composite = (
        decoded_pixels[:, :, :3].astype(np.float32)
        * decoded_pixels[:, :, 3:4].astype(np.float32)
        / 255.0
    )
    composite_difference = np.abs(decoded_composite - source_composite)
    assert float(composite_difference.mean()) < 2.0
    assert float(composite_difference.max()) < 24.0
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        probe = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "stream_tags=alpha_mode",
                "-of",
                "default=noprint_wrappers=1",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "alpha_mode=1" in probe.stdout
