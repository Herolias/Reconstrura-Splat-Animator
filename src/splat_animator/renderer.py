from __future__ import annotations

import math
import sys

import moderngl
import numpy as np

from .camera import (
    _look_at,
    _orbit_eye,
    _perspective,
    _rotation_about_axis,
    _translation,
    _up_vector,
    camera_basis,
    camera_target_offset,
    rotation_center_offset,
)
from .io import SceneData
from .models import AnimationSettings
from .shader_sources import (
    _BACKGROUND_FRAGMENT_SHADER,
    _COMPOSITE_VERTEX_SHADER,
    _FRAGMENT_SHADER,
    _UNPREMULTIPLY_FRAGMENT_SHADER,
    _VERTEX_SHADER,
)
from .video import RenderCancelled, _codec_arguments, expected_extension, render_video

__all__ = [
    "GpuRenderer",
    "RenderCancelled",
    "_codec_arguments",
    "camera_basis",
    "camera_target_offset",
    "create_context",
    "expected_extension",
    "probe_renderer",
    "render_video",
    "rotation_center_offset",
]


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
        self.program["u_camera_position"].value = tuple(float(value) for value in source_camera)
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
