from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import moderngl
import numpy as np

from .io import SceneData
from .models import AnimationSettings

_VERTEX_SHADER = r"""
#version 330

in vec2 in_corner;

uniform sampler2D u_scene_data;
uniform sampler2D u_sh_data;
uniform isampler2D u_draw_order;
uniform int u_scene_texture_width;
uniform int u_sh_texture_width;
uniform int u_sh_coefficient_count;
uniform int u_sh_degree;
uniform int u_order_texture_width;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_projection;
uniform vec2 u_viewport;
uniform vec2 u_focal;
uniform float u_near;
uniform float u_far;

uniform vec3 u_up;
uniform float u_height_min;
uniform float u_height_max;
uniform float u_scan_reverse;
uniform float u_scan_progress;
uniform float u_scan_feather;
uniform float u_rep_source;
uniform float u_rep_target;
uniform int u_transition_effect;
uniform vec3 u_scene_center;
uniform float u_scene_radius;
uniform vec3 u_camera_position;

uniform float u_point_radius;
uniform float u_point_opacity;
uniform float u_splat_scale;
uniform float u_splat_opacity;
uniform float u_min_splat_pixels;
uniform float u_max_splat_pixels;
uniform float u_pixel_scale;

out vec2 v_local;
out vec3 v_color;
out float v_opacity;
out float v_depth;

vec4 fetch_scene(int linear_index) {
    ivec2 location = ivec2(
        linear_index % u_scene_texture_width,
        linear_index / u_scene_texture_width
    );
    return texelFetch(u_scene_data, location, 0);
}

vec3 fetch_sh(int point_index, int coefficient_index) {
    int linear_index = point_index * u_sh_coefficient_count + coefficient_index;
    ivec2 location = ivec2(
        linear_index % u_sh_texture_width,
        linear_index / u_sh_texture_width
    );
    return texelFetch(u_sh_data, location, 0).rgb;
}

vec3 evaluate_sh_color(int point_index, vec3 position, vec3 dc_color) {
    if (u_sh_degree == 0) {
        return dc_color;
    }

    vec3 direction = position - u_camera_position;
    direction /= max(length(direction), 1e-7);
    float x = direction.x;
    float y = direction.y;
    float z = direction.z;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;

    vec3 color = dc_color
        - 0.4886025119029199 * y * fetch_sh(point_index, 0)
        + 0.4886025119029199 * z * fetch_sh(point_index, 1)
        - 0.4886025119029199 * x * fetch_sh(point_index, 2);
    if (u_sh_degree > 1) {
        color +=
            1.0925484305920792 * x * y * fetch_sh(point_index, 3)
            - 1.0925484305920792 * y * z * fetch_sh(point_index, 4)
            + 0.31539156525252005 * (2.0 * zz - xx - yy)
                * fetch_sh(point_index, 5)
            - 1.0925484305920792 * x * z * fetch_sh(point_index, 6)
            + 0.5462742152960396 * (xx - yy) * fetch_sh(point_index, 7);
    }
    if (u_sh_degree > 2) {
        color +=
            -0.5900435899266435 * y * (3.0 * xx - yy)
                * fetch_sh(point_index, 8)
            + 2.890611442640554 * x * y * z * fetch_sh(point_index, 9)
            - 0.4570457994644658 * y * (4.0 * zz - xx - yy)
                * fetch_sh(point_index, 10)
            + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                * fetch_sh(point_index, 11)
            - 0.4570457994644658 * x * (4.0 * zz - xx - yy)
                * fetch_sh(point_index, 12)
            + 1.445305721320277 * z * (xx - yy) * fetch_sh(point_index, 13)
            - 0.5900435899266435 * x * (xx - 3.0 * yy)
                * fetch_sh(point_index, 14);
    }
    return max(color, vec3(0.0));
}

int fetch_draw_index(int linear_index) {
    ivec2 location = ivec2(
        linear_index % u_order_texture_width,
        linear_index / u_order_texture_width
    );
    return texelFetch(u_draw_order, location, 0).r;
}

void main() {
    int point_index = fetch_draw_index(gl_InstanceID);
    vec4 position_opacity = fetch_scene(point_index * 4);
    vec4 color_data = fetch_scene(point_index * 4 + 1);
    vec4 covariance_a_data = fetch_scene(point_index * 4 + 2);
    vec4 covariance_b_data = fetch_scene(point_index * 4 + 3);
    vec3 in_position = position_opacity.xyz;
    float in_opacity = position_opacity.w;
    vec3 in_color = evaluate_sh_color(point_index, in_position, color_data.xyz);
    vec3 in_covariance_a = covariance_a_data.xyz;
    vec3 in_covariance_b = covariance_b_data.xyz;

    vec4 camera_h = u_view * u_model * vec4(in_position, 1.0);
    vec3 camera = camera_h.xyz;
    float depth = -camera.z;
    vec4 clip = u_projection * camera_h;

    float height_range = max(u_height_max - u_height_min, 1e-7);
    float height = dot(in_position, u_up);
    float scan_coordinate = clamp((u_height_max - height) / height_range, 0.0, 1.0);
    float transition_coordinate = scan_coordinate;
    if (u_transition_effect != 0) {
        vec3 normalized_position =
            (in_position - u_scene_center) / max(u_scene_radius, 1e-7);
        if (u_transition_effect == 1) {
            // A spherical reveal feels natural for isolated objects and statues.
            transition_coordinate = clamp(length(normalized_position), 0.0, 1.0);
        } else if (u_transition_effect == 4) {
            // Stable object-space noise: individual splats never flicker over time.
            transition_coordinate = fract(sin(dot(
                normalized_position,
                vec3(127.1, 311.7, 74.7)
            )) * 43758.5453);
        } else {
            vec3 reference = abs(u_up.x) < 0.9
                ? vec3(1.0, 0.0, 0.0)
                : vec3(0.0, 0.0, 1.0);
            vec3 horizontal_a = normalize(reference - u_up * dot(reference, u_up));
            vec3 horizontal_b = cross(u_up, horizontal_a);
            float horizontal_x = dot(normalized_position, horizontal_a);
            float horizontal_y = dot(normalized_position, horizontal_b);
            if (u_transition_effect == 2) {
                // Two low frequencies avoid a mechanically straight edge.
                transition_coordinate = clamp(
                    scan_coordinate +
                    0.075 * sin(horizontal_x * 11.0 + horizontal_y * 3.0) +
                    0.035 * sin(horizontal_y * 17.0 - horizontal_x * 2.0),
                    0.0,
                    1.0
                );
            } else {
                float angle = atan(horizontal_y, horizontal_x) / 6.28318530718 + 0.5;
                transition_coordinate = fract(angle + scan_coordinate * 0.72);
            }
        }
    }
    if (u_scan_reverse > 0.5) {
        transition_coordinate = 1.0 - transition_coordinate;
    }
    float feather = max(u_scan_feather, 0.0001);
    float wave = u_scan_progress * (1.0 + 2.0 * feather) - feather;
    float local_progress = smoothstep(
        transition_coordinate - feather,
        transition_coordinate + feather,
        wave
    );
    float representation = mix(u_rep_source, u_rep_target, local_progress);

    float point_sigma = max(u_point_radius / 2.5, 0.15 * u_pixel_scale);
    vec2 pixel_offset = in_corner * point_sigma;

    if (depth <= u_near || depth >= u_far || clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_opacity = 0.0;
    } else {
        // Pure point-cloud frames skip all covariance projection work. The
        // branch is coherent for the whole draw outside an active transition.
        if (representation > 0.0001) {
            mat3 covariance = mat3(
                vec3(in_covariance_a.x, in_covariance_a.y, in_covariance_a.z),
                vec3(in_covariance_a.y, in_covariance_b.x, in_covariance_b.y),
                vec3(in_covariance_a.z, in_covariance_b.y, in_covariance_b.z)
            );
            mat3 camera_rotation = mat3(u_view * u_model);
            mat3 camera_covariance =
                camera_rotation * covariance * transpose(camera_rotation);

            float safe_depth = max(depth, u_near);
            vec3 jacobian_x = vec3(
                u_focal.x / safe_depth,
                0.0,
                u_focal.x * camera.x / (safe_depth * safe_depth)
            );
            vec3 jacobian_y = vec3(
                0.0,
                u_focal.y / safe_depth,
                u_focal.y * camera.y / (safe_depth * safe_depth)
            );
            float cov_xx = dot(jacobian_x, camera_covariance * jacobian_x);
            float cov_xy = dot(jacobian_x, camera_covariance * jacobian_y);
            float cov_yy = dot(jacobian_y, camera_covariance * jacobian_y);

            float antialias_variance = 0.09 * u_pixel_scale * u_pixel_scale;
            cov_xx = max(
                cov_xx * u_splat_scale * u_splat_scale + antialias_variance,
                0.0
            );
            cov_xy = cov_xy * u_splat_scale * u_splat_scale;
            cov_yy = max(
                cov_yy * u_splat_scale * u_splat_scale + antialias_variance,
                0.0
            );
            float trace = cov_xx + cov_yy;
            float discriminant = sqrt(max(
                (cov_xx - cov_yy) * (cov_xx - cov_yy) + 4.0 * cov_xy * cov_xy,
                0.0
            ));
            float lambda_major = max(0.5 * (trace + discriminant), 0.0);
            float lambda_minor = max(0.5 * (trace - discriminant), 0.0);
            float sigma_major = clamp(
                sqrt(lambda_major),
                u_min_splat_pixels,
                u_max_splat_pixels
            );
            float sigma_minor = clamp(
                sqrt(lambda_minor),
                u_min_splat_pixels,
                u_max_splat_pixels
            );

            vec2 major_axis;
            if (abs(cov_xy) > 0.00001) {
                major_axis = normalize(vec2(cov_xy, lambda_major - cov_xx));
            } else {
                major_axis = cov_xx >= cov_yy ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
            }
            vec2 minor_axis = vec2(-major_axis.y, major_axis.x);
            vec2 gaussian_offset =
                major_axis * in_corner.x * sigma_major +
                minor_axis * in_corner.y * sigma_minor;
            pixel_offset = mix(pixel_offset, gaussian_offset, representation);
        }
        vec2 ndc = clip.xy / clip.w;
        ndc += pixel_offset * (2.0 / u_viewport);
        gl_Position = vec4(ndc * clip.w, clip.z, clip.w);
        v_opacity = mix(u_point_opacity, in_opacity * u_splat_opacity, representation);
    }
    v_local = in_corner;
    v_color = in_color;
    v_depth = clamp((depth - u_near) / max(u_far - u_near, 0.0001), 0.0, 1.0);
}
"""


_FRAGMENT_SHADER = r"""
#version 330

in vec2 v_local;
in vec3 v_color;
in float v_opacity;
uniform float u_exposure;

layout(location = 0) out vec4 out_color;

void main() {
    float power = -0.5 * dot(v_local, v_local);
    if (power < -8.0) {
        discard;
    }
    float alpha = min(0.999, exp(power) * v_opacity);
    if (alpha < (1.0 / 255.0)) {
        discard;
    }
    out_color = vec4(clamp(v_color * exp2(u_exposure), 0.0, 1.0), alpha);
}
"""


_COMPOSITE_VERTEX_SHADER = r"""
#version 330

out vec2 v_uv;

void main() {
    vec2 position = vec2(
        (gl_VertexID == 1) ? 3.0 : -1.0,
        (gl_VertexID == 2) ? 3.0 : -1.0
    );
    v_uv = position * 0.5 + 0.5;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""


_BACKGROUND_FRAGMENT_SHADER = r"""
#version 330

uniform vec3 u_background;
uniform float u_background_gradient;

in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 centered = v_uv - vec2(0.5);
    float glow = 1.0 - smoothstep(0.0, 0.72, length(centered));
    vec3 background = u_background * mix(
        1.0 - u_background_gradient,
        1.0 + u_background_gradient,
        glow
    );
    out_color = vec4(clamp(background, 0.0, 1.0), 1.0);
}
"""


_UNPREMULTIPLY_FRAGMENT_SHADER = r"""
#version 330

uniform sampler2D u_composite;

in vec2 v_uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec4 composite = texture(u_composite, v_uv);
    vec3 straight_color = composite.a > (1.0 / 65535.0)
        ? composite.rgb / composite.a
        : vec3(0.0);
    out_color = vec4(clamp(straight_color, 0.0, 1.0), composite.a);
}
"""


class RenderCancelled(RuntimeError):
    pass


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


def _hex_color(value: str) -> tuple[float, float, float]:
    stripped = value.strip().lstrip("#")
    if len(stripped) == 3:
        stripped = "".join(character * 2 for character in stripped)
    if len(stripped) != 6:
        return 0.03, 0.04, 0.065
    try:
        return tuple(int(stripped[index : index + 2], 16) / 255.0 for index in (0, 2, 4))
    except ValueError:
        return 0.03, 0.04, 0.065


def create_context() -> moderngl.Context:
    errors: list[Exception] = []
    backends: tuple[str | None, ...] = (
        ("egl", None) if sys.platform.startswith("linux") else (None,)
    )
    for backend in backends:
        try:
            if backend:
                return moderngl.create_standalone_context(require=330, backend=backend)
            return moderngl.create_standalone_context(require=330)
        except Exception as exc:  # pragma: no cover - platform-specific failure detail
            errors.append(exc)
    detail = "; ".join(str(error) for error in errors)
    raise RuntimeError(f"Could not create an OpenGL 3.3 rendering context: {detail}")


def probe_renderer() -> str:
    context = create_context()
    try:
        return str(context.info.get("GL_RENDERER", "OpenGL 3.3"))
    finally:
        context.release()


class GpuRenderer:
    def __init__(self, scene: SceneData) -> None:
        self.scene = scene
        sh_coefficients = scene.sh_coefficients
        if sh_coefficients is not None and (
            sh_coefficients.ndim != 3
            or sh_coefficients.shape[0] != scene.count
            or sh_coefficients.shape[2] != 3
        ):
            raise ValueError("SH coefficients must have shape (point count, coefficient count, 3)")
        self.sh_coefficient_count = (
            min(int(sh_coefficients.shape[1]), 15) if sh_coefficients is not None else 0
        )
        self.sh_degree = max(
            (
                degree
                for degree in range(1, 4)
                if (degree + 1) ** 2 - 1 <= self.sh_coefficient_count
            ),
            default=0,
        )
        self.context = create_context()
        self.program = self.context.program(
            vertex_shader=_VERTEX_SHADER,
            fragment_shader=_FRAGMENT_SHADER,
        )
        self.background_program = self.context.program(
            vertex_shader=_COMPOSITE_VERTEX_SHADER,
            fragment_shader=_BACKGROUND_FRAGMENT_SHADER,
        )
        self.unpremultiply_program = self.context.program(
            vertex_shader=_COMPOSITE_VERTEX_SHADER,
            fragment_shader=_UNPREMULTIPLY_FRAGMENT_SHADER,
        )

        corners = np.array(
            [[-3.0, -3.0], [3.0, -3.0], [-3.0, 3.0], [3.0, 3.0]],
            dtype="f4",
        )
        self.corner_buffer = self.context.buffer(corners.tobytes())
        self.vertex_array = self.context.vertex_array(
            self.program,
            [(self.corner_buffer, "2f", "in_corner")],
        )
        self.background_array = self.context.vertex_array(self.background_program, [])
        self.unpremultiply_array = self.context.vertex_array(self.unpremultiply_program, [])

        maximum_texture_size = int(self.context.info.get("GL_MAX_TEXTURE_SIZE", 16384))
        self.scene_texture_width = min(4096, maximum_texture_size)
        scene_texel_count = scene.count * 4
        scene_texture_height = math.ceil(scene_texel_count / self.scene_texture_width)
        if scene_texture_height > maximum_texture_size:
            raise RuntimeError(
                f"Scene needs a {self.scene_texture_width} x {scene_texture_height} data "
                f"texture, but this GPU supports at most {maximum_texture_size} x "
                f"{maximum_texture_size}"
            )
        scene_storage = np.zeros(
            (scene_texture_height * self.scene_texture_width, 4),
            dtype=np.float32,
        )
        scene_records = scene_storage[:scene_texel_count].reshape(scene.count, 4, 4)
        scene_records[:, 0, :3] = scene.positions
        scene_records[:, 0, 3] = scene.opacity[:, 0]
        scene_records[:, 1, :3] = scene.colors
        scene_records[:, 2, :3] = scene.covariance_a
        scene_records[:, 3, :3] = scene.covariance_b
        self.scene_texture = self.context.texture(
            (self.scene_texture_width, scene_texture_height),
            4,
            data=scene_storage.tobytes(),
            dtype="f4",
        )
        self.scene_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.scene_texture.repeat_x = False
        self.scene_texture.repeat_y = False

        if self.sh_degree:
            self.sh_texture_width = min(4096, maximum_texture_size)
            sh_texel_count = scene.count * self.sh_coefficient_count
            sh_texture_height = math.ceil(sh_texel_count / self.sh_texture_width)
            if sh_texture_height > maximum_texture_size:
                raise RuntimeError(
                    f"Scene needs a {self.sh_texture_width} x {sh_texture_height} SH data "
                    f"texture, but this GPU supports at most {maximum_texture_size} x "
                    f"{maximum_texture_size}"
                )
            sh_storage = np.zeros(
                (sh_texture_height * self.sh_texture_width, 3),
                dtype=np.float32,
            )
            assert sh_coefficients is not None
            sh_storage[:sh_texel_count] = sh_coefficients[
                :, : self.sh_coefficient_count, :
            ].reshape(-1, 3)
            sh_texture_size = (self.sh_texture_width, sh_texture_height)
        else:
            self.sh_texture_width = 1
            sh_storage = np.zeros((1, 3), dtype=np.float32)
            sh_texture_size = (1, 1)
        self.sh_texture = self.context.texture(
            sh_texture_size,
            3,
            data=sh_storage.tobytes(),
            dtype="f4",
        )
        self.sh_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.sh_texture.repeat_x = False
        self.sh_texture.repeat_y = False

        self.order_texture_width = min(4096, maximum_texture_size)
        order_texture_height = math.ceil(scene.count / self.order_texture_width)
        self.order_storage = np.zeros(
            order_texture_height * self.order_texture_width,
            dtype=np.int32,
        )
        self.order_storage[: scene.count] = np.arange(scene.count, dtype=np.int32)
        self.order_texture = self.context.texture(
            (self.order_texture_width, order_texture_height),
            1,
            data=self.order_storage.tobytes(),
            dtype="i4",
        )
        self.order_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.order_texture.repeat_x = False
        self.order_texture.repeat_y = False
        self._last_depth_row: np.ndarray | None = None

        self.render_texture: moderngl.Texture | None = None
        self.render_framebuffer: moderngl.Framebuffer | None = None
        self.alpha_texture: moderngl.Texture | None = None
        self.alpha_framebuffer: moderngl.Framebuffer | None = None
        self.target_size = (0, 0)
        self.height_bounds = {
            axis: tuple(
                float(item) for item in np.quantile(scene.positions[:, index], (0.0025, 0.9975))
            )
            for index, axis in enumerate(("x", "y", "z"))
        }

    @property
    def renderer_name(self) -> str:
        return str(self.context.info.get("GL_RENDERER", "OpenGL 3.3"))

    def _release_targets(self) -> None:
        for value in (
            self.alpha_framebuffer,
            self.alpha_texture,
            self.render_framebuffer,
            self.render_texture,
        ):
            if value is not None:
                value.release()
        self.render_texture = None
        self.render_framebuffer = None
        self.alpha_texture = None
        self.alpha_framebuffer = None
        self.target_size = (0, 0)

    def _ensure_targets(self, width: int, height: int, *, alpha_output: bool = False) -> None:
        if self.target_size != (width, height):
            self._release_targets()
            self.render_texture = self.context.texture((width, height), 4, dtype="f2")
            self.render_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.render_framebuffer = self.context.framebuffer(
                color_attachments=[self.render_texture]
            )
            self.target_size = (width, height)
        if alpha_output and self.alpha_framebuffer is None:
            self.alpha_texture = self.context.texture((width, height), 4, dtype="f1")
            self.alpha_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.alpha_framebuffer = self.context.framebuffer(
                color_attachments=[self.alpha_texture]
            )

    @staticmethod
    def _write_matrix(uniform: moderngl.Uniform, matrix: np.ndarray) -> None:
        uniform.write(np.ascontiguousarray(matrix.T, dtype="f4").tobytes())

    def _set_uniforms(
        self,
        settings: AnimationSettings,
        seconds: float,
        *,
        sort_depth: bool = True,
    ) -> None:
        up = _up_vector(settings.up_axis)
        angle = settings.angle_at(seconds)
        distance = self.scene.radius * settings.distance_scale
        target_offset = camera_target_offset(settings, seconds)
        pivot_offset = rotation_center_offset(settings, seconds, target=target_offset)
        target = self.scene.radius * target_offset
        pivot = self.scene.radius * pivot_offset
        if settings.rotation_mode == "camera_orbit":
            model = _translation(-self.scene.center)
            camera_angle = angle
            eye_offset = _orbit_eye(up, camera_angle, settings.elevation, distance)
            eye = pivot + eye_offset
        else:
            model = (
                _translation(pivot)
                @ _rotation_about_axis(up, angle)
                @ _translation(-pivot)
                @ _translation(-self.scene.center)
            )
            camera_angle = 0.0
            eye_offset = _orbit_eye(up, camera_angle, settings.elevation, distance)
            eye = target + eye_offset
        view = _look_at(eye, target, up, -eye_offset)
        source_camera = np.linalg.solve(
            model,
            np.array([eye[0], eye[1], eye[2], 1.0], dtype=np.float32),
        )[:3]
        near = max(min(self.scene.radius * 0.01, distance * 0.02), 1e-6)
        far = max(
            float(np.linalg.norm(eye))
            + self.scene.radius * 4.0
            + 2.0 * float(np.linalg.norm(pivot)),
            near + 1.0,
        )
        projection = _perspective(settings.fov, settings.width / settings.height, near, far)
        focal_y = settings.height / (2.0 * math.tan(math.radians(settings.fov) * 0.5))
        representation = settings.representation_at(seconds)
        height_min, height_max = self.height_bounds[settings.up_axis[-1]]
        if settings.up_axis.startswith("-"):
            height_min, height_max = -height_max, -height_min

        self._write_matrix(self.program["u_model"], model)
        self._write_matrix(self.program["u_view"], view)
        self._write_matrix(self.program["u_projection"], projection)
        self.program["u_viewport"].value = (settings.width, settings.height)
        self.program["u_focal"].value = (focal_y, focal_y)
        self.program["u_near"].value = near
        self.program["u_far"].value = far
        self.program["u_up"].value = tuple(float(value) for value in up)
        self.program["u_height_min"].value = height_min
        self.program["u_height_max"].value = height_max
        self.program["u_scan_reverse"].value = float(settings.scan_direction == "bottom_to_top")
        self.program["u_scan_progress"].value = representation.progress
        self.program["u_scan_feather"].value = settings.scan_feather
        self.program["u_rep_source"].value = representation.source
        self.program["u_rep_target"].value = representation.target
        transition_effect = (
            settings.transition_effect if settings.transformation_enabled else "sweep"
        )
        self.program["u_transition_effect"].value = {
            "sweep": 0,
            "radial": 1,
            "wave": 2,
            "spiral": 3,
            "dissolve": 4,
        }[transition_effect]
        self.program["u_scene_center"].value = tuple(float(value) for value in self.scene.center)
        self.program["u_scene_radius"].value = self.scene.radius
        self.program["u_camera_position"].value = tuple(
            float(value) for value in source_camera
        )
        pixel_scale = min(settings.width, settings.height) / 1080.0
        self.program["u_pixel_scale"].value = pixel_scale
        self.program["u_point_radius"].value = settings.point_radius * pixel_scale
        self.program["u_point_opacity"].value = settings.point_opacity
        self.program["u_splat_scale"].value = settings.splat_scale
        self.program["u_splat_opacity"].value = settings.splat_opacity
        self.program["u_min_splat_pixels"].value = settings.min_splat_pixels * pixel_scale
        self.program["u_max_splat_pixels"].value = settings.max_splat_pixels * pixel_scale
        self.program["u_exposure"].value = settings.exposure
        self.program["u_scene_data"].value = 0
        self.program["u_draw_order"].value = 1
        self.program["u_sh_data"].value = 2
        self.program["u_scene_texture_width"].value = self.scene_texture_width
        self.program["u_sh_texture_width"].value = self.sh_texture_width
        self.program["u_sh_coefficient_count"].value = self.sh_coefficient_count
        self.program["u_sh_degree"].value = self.sh_degree
        self.program["u_order_texture_width"].value = self.order_texture_width

        depth_row = np.asarray((view @ model)[2], dtype=np.float32)
        needs_initial_sort = self._last_depth_row is None
        view_changed = not needs_initial_sort and not np.array_equal(
            depth_row[:3], self._last_depth_row[:3]
        )
        # Interactive previews may temporarily retain the last exact order. An
        # initial frame is always sorted, and exports keep sort_depth enabled.
        if needs_initial_sort or (sort_depth and view_changed):
            # Camera translation adds the same value to every depth and cannot
            # change ordering. Ignoring it avoids needless re-sorts for pan,
            # dolly, and true fly navigation.
            camera_z = self.scene.positions @ depth_row[:3]
            # Equal-depth source order has no physical significance, so an
            # unstable exact sort is both valid and substantially faster here.
            self.order_storage[: self.scene.count] = np.argsort(
                camera_z,
                kind="quicksort",
            ).astype(np.int32)
            self.order_texture.write(self.order_storage.tobytes())
            self._last_depth_row = depth_row.copy()

    def _render_frame(
        self,
        settings: AnimationSettings,
        seconds: float,
        *,
        transparent: bool,
        sort_depth: bool = True,
    ) -> bytes:
        settings.validate()
        self._ensure_targets(
            settings.width,
            settings.height,
            alpha_output=transparent,
        )
        assert self.render_framebuffer is not None

        self._set_uniforms(settings, seconds, sort_depth=sort_depth)
        self.render_framebuffer.use()
        self.context.viewport = (0, 0, settings.width, settings.height)
        self.context.disable(moderngl.BLEND)
        self.context.disable(moderngl.DEPTH_TEST)
        if transparent:
            self.render_framebuffer.clear(0.0, 0.0, 0.0, 0.0)
        else:
            self.background_program["u_background"].value = _hex_color(settings.background)
            self.background_program["u_background_gradient"].value = settings.background_gradient
            self.background_array.render(mode=moderngl.TRIANGLES, vertices=3)

        self.scene_texture.use(location=0)
        self.order_texture.use(location=1)
        self.sh_texture.use(location=2)
        self.context.enable(moderngl.BLEND)
        self.context.blend_func = (
            moderngl.SRC_ALPHA,
            moderngl.ONE_MINUS_SRC_ALPHA,
            moderngl.ONE,
            moderngl.ONE_MINUS_SRC_ALPHA,
        )
        self.vertex_array.render(
            mode=moderngl.TRIANGLE_STRIP,
            vertices=4,
            instances=self.scene.count,
        )
        self.context.disable(moderngl.BLEND)

        if transparent:
            if settings.premultiplied_alpha:
                return self.render_framebuffer.read(components=4, alignment=1, dtype="f1")
            assert self.alpha_framebuffer is not None
            assert self.render_texture is not None
            self.alpha_framebuffer.use()
            self.render_texture.use(location=0)
            self.unpremultiply_program["u_composite"].value = 0
            self.unpremultiply_array.render(mode=moderngl.TRIANGLES, vertices=3)
            return self.alpha_framebuffer.read(components=4, alignment=1, dtype="f1")

        return self.render_framebuffer.read(components=3, alignment=1, dtype="f1")

    def render_frame(
        self,
        settings: AnimationSettings,
        seconds: float,
        *,
        sort_depth: bool = True,
    ) -> bytes:
        """Render opaque RGB or transparent straight/premultiplied RGBA."""
        return self._render_frame(
            settings,
            seconds,
            transparent=settings.transparent_background,
            sort_depth=sort_depth,
        )

    def render_rgb(
        self,
        settings: AnimationSettings,
        seconds: float,
        *,
        sort_depth: bool = True,
    ) -> bytes:
        """Render an opaque RGB frame, retained for API compatibility."""
        return self._render_frame(
            settings,
            seconds,
            transparent=False,
            sort_depth=sort_depth,
        )

    def close(self) -> None:
        self._release_targets()
        for value in (
            self.vertex_array,
            self.background_array,
            self.unpremultiply_array,
            self.corner_buffer,
            self.scene_texture,
            self.sh_texture,
            self.order_texture,
            self.program,
            self.background_program,
            self.unpremultiply_program,
        ):
            value.release()
        self.context.release()

    def __enter__(self) -> GpuRenderer:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _codec_arguments(settings: AnimationSettings) -> list[str]:
    quality = str(settings.quality)
    rate_arguments = (
        ["-b:v", f"{settings.bitrate_mbps:g}M"]
        if settings.bitrate_mbps > 0
        else ["-crf", quality]
    )
    if settings.codec == "h264":
        return [
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            *rate_arguments,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
    if settings.codec == "h265":
        return [
            "-c:v",
            "libx265",
            "-preset",
            "slow",
            *rate_arguments,
            "-pix_fmt",
            "yuv420p",
            "-tag:v",
            "hvc1",
        ]
    if settings.codec == "prores":
        return ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"]
    vp9_rate_arguments = (
        rate_arguments if settings.bitrate_mbps > 0 else [*rate_arguments, "-b:v", "0"]
    )
    arguments = [
        "-c:v",
        "libvpx-vp9",
        *vp9_rate_arguments,
        "-row-mt",
        "1",
        "-pix_fmt",
        "yuva420p" if settings.transparent_background else "yuv420p",
    ]
    if settings.transparent_background:
        arguments.extend(("-auto-alt-ref", "0"))
    return arguments


def expected_extension(codec: str) -> str:
    return {"h264": ".mp4", "h265": ".mp4", "prores": ".mov", "vp9": ".webm"}[codec]


def render_video(
    renderer: GpuRenderer,
    settings: AnimationSettings,
    output: str | Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    settings.validate()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH")

    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    required_suffix = expected_extension(settings.codec)
    if target.suffix.lower() != required_suffix:
        target = target.with_suffix(required_suffix)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}-",
        suffix=target.suffix,
        dir=target.parent,
    )
    os.close(descriptor)
    Path(temporary_name).unlink()
    temporary = Path(temporary_name)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba" if settings.transparent_background else "rgb24",
        "-video_size",
        f"{settings.width}x{settings.height}",
        "-framerate",
        str(settings.fps),
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        (
            "vflip,scale=in_color_matrix=bt709:out_color_matrix=bt709:"
            "in_range=full:out_range=limited,"
            "setparams=range=limited:color_primaries=bt709:"
            "color_trc=bt709:colorspace=bt709"
        ),
        *_codec_arguments(settings),
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame_index in range(settings.frame_count):
            if cancel_event is not None and cancel_event.is_set():
                raise RenderCancelled("Video render cancelled")
            frame = renderer.render_frame(settings, frame_index / settings.fps)
            try:
                process.stdin.write(frame)
            except BrokenPipeError as exc:
                stderr = (
                    process.stderr.read().decode("utf-8", errors="replace")
                    if process.stderr
                    else ""
                )
                process.wait()
                detail = stderr.strip() or "the encoder closed its input unexpectedly"
                raise RuntimeError(f"FFmpeg stopped while receiving frames: {detail}") from exc
            if progress is not None:
                progress(frame_index + 1, settings.frame_count)
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg failed ({return_code}): {stderr.strip()}")
        os.replace(temporary, target)
        return target
    except (Exception, KeyboardInterrupt):
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except BrokenPipeError:
                # FFmpeg may already have exited; preserve the original render
                # or pipe exception instead of masking it during cleanup.
                pass
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if process.stderr is not None:
            process.stderr.close()
