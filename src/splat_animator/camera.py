from __future__ import annotations

import math

import numpy as np

from .models import AnimationSettings


def _normalize(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def _up_vector(axis: str) -> np.ndarray:
    values = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    sign = -1.0 if axis.startswith("-") else 1.0
    return values[axis[-1]] * sign


def _rotation_about_axis(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    angle = math.radians(angle_degrees)
    x, y, z = _normalize(axis)
    cosine, sine = math.cos(angle), math.sin(angle)
    one_minus = 1.0 - cosine
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float32,
    )
    return matrix


def _translation(offset: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = offset
    return matrix


def _camera_axes(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fallback_forward: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = target - eye
    forward = (
        _normalize(fallback_forward)
        if float(np.linalg.norm(direction)) < 1e-8
        else _normalize(direction)
    )
    side = np.cross(forward, up)
    if float(np.linalg.norm(side)) < 1e-8:
        reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(reference, forward))) > 0.95:
            reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        side = np.cross(forward, reference)
    side = _normalize(side)
    camera_up = np.cross(side, forward)
    return side, camera_up, forward


def _look_at(
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fallback_forward: np.ndarray,
) -> np.ndarray:
    side, camera_up, forward = _camera_axes(eye, target, up, fallback_forward)
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, :3] = side
    matrix[1, :3] = camera_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(side, eye)
    matrix[1, 3] = -np.dot(camera_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def _perspective(fov_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    focal = 1.0 / math.tan(math.radians(fov_degrees) * 0.5)
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def _orbit_eye(
    up: np.ndarray,
    angle_degrees: float,
    elevation_degrees: float,
    distance: float,
) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(reference, up))) > 0.95:
        reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    horizontal_a = _normalize(reference - up * np.dot(reference, up))
    horizontal_b = _normalize(np.cross(up, horizontal_a))
    azimuth = math.radians(angle_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = horizontal_a * math.cos(azimuth) + horizontal_b * math.sin(azimuth)
    return distance * (horizontal * math.cos(elevation) + up * math.sin(elevation))


def _nominal_camera_basis(
    settings: AnimationSettings,
    seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the old target-centered basis, used to migrate legacy pan."""
    up = _up_vector(settings.up_axis)
    camera_angle = settings.angle_at(seconds) if settings.rotation_mode == "camera_orbit" else 0.0
    eye_offset = _orbit_eye(up, camera_angle, settings.elevation, 1.0)
    forward = _normalize(-eye_offset)
    right = _normalize(np.cross(forward, up))
    camera_up = _normalize(np.cross(right, forward))
    return right, camera_up, forward


def camera_target_offset(settings: AnimationSettings, seconds: float) -> np.ndarray:
    """Return the saved target in scene-radius units, including old presets."""
    target = np.array(
        [settings.camera_target_x, settings.camera_target_y, settings.camera_target_z],
        dtype=np.float32,
    )
    if settings.camera_pan_x or settings.camera_pan_y:
        right, camera_up, _forward = _nominal_camera_basis(settings, seconds)
        target += right * settings.camera_pan_x + camera_up * settings.camera_pan_y
    return target


def rotation_center_offset(
    settings: AnimationSettings,
    seconds: float,
    *,
    target: np.ndarray | None = None,
) -> np.ndarray:
    """Return the animation pivot in centered scene-radius units."""
    if target is None:
        target = camera_target_offset(settings, seconds)
    mode = settings.rotation_center_mode
    if mode == "automatic":
        return (
            target.copy()
            if settings.rotation_mode == "camera_orbit"
            else np.zeros(3, dtype=np.float32)
        )
    if mode == "target":
        return target.copy()
    if mode == "custom":
        return np.array(
            [
                settings.rotation_center_x,
                settings.rotation_center_y,
                settings.rotation_center_z,
            ],
            dtype=np.float32,
        )
    return np.zeros(3, dtype=np.float32)


def camera_basis(
    settings: AnimationSettings,
    seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return camera right, up, and forward axes in centered world space."""
    up = _up_vector(settings.up_axis)
    target = camera_target_offset(settings, seconds)
    camera_angle = settings.angle_at(seconds) if settings.rotation_mode == "camera_orbit" else 0.0
    eye_offset = _orbit_eye(up, camera_angle, settings.elevation, settings.distance_scale)
    if settings.rotation_mode == "camera_orbit":
        pivot = rotation_center_offset(settings, seconds, target=target)
        eye = pivot + eye_offset
    else:
        eye = target + eye_offset
    return _camera_axes(eye, target, up, -eye_offset)
