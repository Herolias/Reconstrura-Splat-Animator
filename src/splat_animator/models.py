from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RepresentationFrame:
    """The two representations blended by the spatial scan at one frame."""

    source: float
    target: float
    progress: float


@dataclass(frozen=True)
class AnimationSettings:
    duration: float = 12.0
    fps: int = 30
    width: int = 1920
    height: int = 1080
    seamless_loop: bool = False

    rotation_mode: str = "camera_orbit"
    spin_speed: float = 30.0
    spin_direction: str = "clockwise"
    start_angle: float = 0.0
    elevation: float = 12.0
    distance_scale: float = 2.65
    camera_target_x: float = 0.0
    camera_target_y: float = 0.0
    camera_target_z: float = 0.0
    rotation_center_mode: str = "automatic"
    rotation_center_x: float = 0.0
    rotation_center_y: float = 0.0
    rotation_center_z: float = 0.0
    # Kept for preset/CLI compatibility. New camera movement is stored as the
    # world-space target above so it does not rotate when the camera orbits.
    camera_pan_x: float = 0.0
    camera_pan_y: float = 0.0
    fov: float = 48.0

    start_representation: str = "splat"
    transformation_enabled: bool = True
    round_trip: bool = True
    transition_start: float = 2.0
    transition_duration: float = 2.2
    return_hold: float = 2.6
    easing: str = "cinematic"
    transition_effect: str = "sweep"
    scan_direction: str = "top_to_bottom"
    up_axis: str = "-y"
    scan_feather: float = 0.055

    point_radius: float = 0.75
    point_opacity: float = 0.55
    splat_scale: float = 1.0
    splat_opacity: float = 1.0
    min_splat_pixels: float = 0.45
    max_splat_pixels: float = 96.0
    exposure: float = 0.0
    background: str = "#080b11"
    background_gradient: float = 0.16

    codec: str = "h264"
    quality: int = 18

    def validate(self) -> None:
        if not isinstance(self.seamless_loop, bool):
            raise ValueError("Seamless loop must be true or false")
        timing_values = (
            self.duration,
            self.transition_start,
            self.transition_duration,
            self.return_hold,
        )
        if not all(math.isfinite(value) for value in timing_values):
            raise ValueError("Animation timing values must be finite")
        if self.duration <= 0:
            raise ValueError("Video duration must be greater than zero")
        if not isinstance(self.fps, int) or isinstance(self.fps, bool):
            raise ValueError("FPS must be an integer")
        if not 1 <= self.fps <= 240:
            raise ValueError("FPS must be between 1 and 240")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (self.width, self.height)
        ):
            raise ValueError("Video width and height must be integers")
        if self.width < 16 or self.height < 16:
            raise ValueError("Resolution must be at least 16 × 16")
        if self.width % 2 or self.height % 2:
            raise ValueError("Video width and height must be even")
        if self.transition_start < 0 or self.transition_duration <= 0:
            raise ValueError("Transition timing is invalid")
        if self.return_hold < 0:
            raise ValueError("Return hold cannot be negative")
        if self.rotation_mode not in {"camera_orbit", "object_spin"}:
            raise ValueError(f"Unknown rotation mode: {self.rotation_mode}")
        if self.rotation_center_mode not in {"automatic", "scene", "target", "custom"}:
            raise ValueError(f"Unknown rotation center mode: {self.rotation_center_mode}")
        if self.spin_direction not in {"clockwise", "counter_clockwise"}:
            raise ValueError(f"Unknown spin direction: {self.spin_direction}")
        motion_values = (self.spin_speed, self.start_angle, self.elevation)
        if not all(math.isfinite(value) for value in motion_values):
            raise ValueError("Camera motion values must be finite")
        if self.spin_speed < 0:
            raise ValueError("Spin speed cannot be negative")
        if not -89.9 <= self.elevation <= 89.9:
            raise ValueError("Camera elevation must stay between -89.9 and 89.9 degrees")
        if not math.isfinite(self.distance_scale) or self.distance_scale <= 0:
            raise ValueError("Camera distance must be greater than zero")
        if not math.isfinite(self.fov) or not 1.0 <= self.fov < 179.0:
            raise ValueError("Field of view must be between 1 and 179 degrees")
        camera_values = (
            self.camera_target_x,
            self.camera_target_y,
            self.camera_target_z,
            self.rotation_center_x,
            self.rotation_center_y,
            self.rotation_center_z,
            self.camera_pan_x,
            self.camera_pan_y,
        )
        if not all(math.isfinite(value) for value in camera_values):
            raise ValueError("Camera offsets must be finite")
        if self.start_representation not in {"splat", "point"}:
            raise ValueError(f"Unknown starting representation: {self.start_representation}")
        if not isinstance(self.transformation_enabled, bool):
            raise ValueError("Transformation enabled must be true or false")
        if not isinstance(self.round_trip, bool):
            raise ValueError("Round trip must be true or false")
        if self.easing not in {"linear", "smooth", "cinematic"}:
            raise ValueError(f"Unknown easing: {self.easing}")
        if self.transition_effect not in {"sweep", "radial", "wave", "spiral", "dissolve"}:
            raise ValueError(f"Unknown transition effect: {self.transition_effect}")
        if self.up_axis not in {"x", "-x", "y", "-y", "z", "-z"}:
            raise ValueError(f"Unknown up axis: {self.up_axis}")
        if self.scan_direction not in {"top_to_bottom", "bottom_to_top"}:
            raise ValueError(f"Unknown scan direction: {self.scan_direction}")
        if not math.isfinite(self.scan_feather) or self.scan_feather <= 0:
            raise ValueError("Transition edge softness must be greater than zero")
        visual_values = (
            self.point_radius,
            self.point_opacity,
            self.splat_scale,
            self.splat_opacity,
            self.min_splat_pixels,
            self.max_splat_pixels,
            self.exposure,
            self.background_gradient,
        )
        if not all(math.isfinite(value) for value in visual_values):
            raise ValueError("Appearance values must be finite")
        if self.point_radius <= 0 or self.splat_scale <= 0:
            raise ValueError("Point and splat sizes must be greater than zero")
        if self.point_opacity < 0 or self.splat_opacity < 0:
            raise ValueError("Point and splat opacity cannot be negative")
        if self.min_splat_pixels < 0 or self.max_splat_pixels < self.min_splat_pixels:
            raise ValueError("Splat pixel limits are invalid")
        if not isinstance(self.background, str):
            raise ValueError("Background must be a hex color string")
        background = self.background.removeprefix("#")
        if len(background) not in {3, 6} or any(
            character not in "0123456789abcdefABCDEF" for character in background
        ):
            raise ValueError("Background must be a 3- or 6-digit hex color")
        if self.codec not in {"h264", "h265", "prores", "vp9"}:
            raise ValueError(f"Unknown codec: {self.codec}")
        if not isinstance(self.quality, int) or isinstance(self.quality, bool):
            raise ValueError("Codec quality must be an integer")
        maximum_quality = 63 if self.codec == "vp9" else 51
        if not 0 <= self.quality <= maximum_quality:
            raise ValueError(
                f"{self.codec.upper()} quality must be between 0 and {maximum_quality}"
            )

    @property
    def frame_count(self) -> int:
        return max(1, round(self.duration * self.fps))

    @property
    def return_start(self) -> float:
        return self.transition_start + self.transition_duration + self.return_hold

    @property
    def transition_end(self) -> float:
        if not self.transformation_enabled:
            return 0.0
        end = self.transition_start + self.transition_duration
        if self.round_trip:
            end = self.return_start + self.transition_duration
        return end

    @property
    def spin_sign(self) -> float:
        return -1.0 if self.spin_direction == "clockwise" else 1.0

    def angle_at(self, seconds: float) -> float:
        return self.start_angle + self.spin_sign * self.spin_speed * seconds

    def resolved_for_loop(self) -> AnimationSettings:
        """Return frame-aligned settings whose end joins the first frame.

        An enabled seamless transform must return to its starting
        representation, so loop mode resolves it to a completed round trip. A
        static representation needs only a complete rotation. Rotation is
        fitted to an integer turn while minimizing the combined relative
        duration and speed changes. The original instance remains untouched for
        editable presets.
        """
        if not self.seamless_loop:
            return self

        minimum_duration = (
            self.transition_start + self.transition_duration * 2.0 + self.return_hold
            if self.transformation_enabled
            else 1.0 / self.fps
        )
        minimum_frames = max(1, math.ceil(minimum_duration * self.fps - 1e-9))
        chosen_frames = max(minimum_frames, round(self.duration * self.fps))
        speed = abs(self.spin_speed)
        if speed < 1e-8:
            return replace(
                self,
                duration=chosen_frames / self.fps,
                spin_speed=0.0,
                round_trip=True if self.transformation_enabled else self.round_trip,
            )

        base_duration = max(self.duration, minimum_duration, 1.0 / self.fps)
        turn_estimates = (
            speed * base_duration / 360.0,
            speed * minimum_duration / 360.0,
        )
        turn_counts: set[int] = set()
        for estimate in turn_estimates:
            center = max(1, round(estimate))
            turn_counts.update(range(max(1, center - 3), center + 4))

        best: tuple[tuple[float, float, float], int, float] | None = None
        for turns in turn_counts:
            duration_at_current_speed = turns * 360.0 / speed
            balanced_duration = math.sqrt(base_duration * duration_at_current_speed)
            duration_candidates = (
                base_duration,
                minimum_duration,
                duration_at_current_speed,
                balanced_duration,
            )
            for candidate_duration in duration_candidates:
                center_frame = max(minimum_frames, round(candidate_duration * self.fps))
                for frames in range(max(minimum_frames, center_frame - 2), center_frame + 3):
                    resolved_duration = frames / self.fps
                    resolved_speed = turns * 360.0 / resolved_duration
                    duration_change = abs(resolved_duration - self.duration) / max(
                        self.duration, 1.0 / self.fps
                    )
                    speed_change = abs(resolved_speed - speed) / speed
                    score = (
                        duration_change * duration_change + speed_change * speed_change,
                        duration_change,
                        speed_change,
                    )
                    candidate = (score, frames, resolved_speed)
                    if best is None or candidate < best:
                        best = candidate

        assert best is not None
        return replace(
            self,
            duration=best[1] / self.fps,
            spin_speed=best[2],
            round_trip=True if self.transformation_enabled else self.round_trip,
        )

    def _ease(self, value: float) -> float:
        value = min(1.0, max(0.0, value))
        if self.easing == "linear":
            return value
        if self.easing == "smooth":
            return value * value * (3.0 - 2.0 * value)
        # Quintic smootherstep has zero first and second derivatives at both ends.
        return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)

    def representation_at(self, seconds: float) -> RepresentationFrame:
        start = 1.0 if self.start_representation == "splat" else 0.0
        if not self.transformation_enabled:
            return RepresentationFrame(start, start, 0.0)
        other = 1.0 - start
        first_end = self.transition_start + self.transition_duration

        if seconds < self.transition_start:
            return RepresentationFrame(start, start, 0.0)
        if seconds < first_end:
            raw = (seconds - self.transition_start) / self.transition_duration
            return RepresentationFrame(start, other, self._ease(raw))
        if not self.round_trip:
            return RepresentationFrame(other, other, 0.0)
        if seconds < self.return_start:
            return RepresentationFrame(other, other, 0.0)
        if seconds < self.return_start + self.transition_duration:
            raw = (seconds - self.return_start) / self.transition_duration
            return RepresentationFrame(other, start, self._ease(raw))
        return RepresentationFrame(start, start, 0.0)

    def with_updates(self, **updates: Any) -> AnimationSettings:
        value = replace(self, **updates)
        value.validate()
        return value

    def to_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> AnimationSettings:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Animation preset must contain a JSON object")
        known = {field.name for field in fields(cls)}
        settings = cls(**{key: value for key, value in payload.items() if key in known})
        try:
            settings.validate()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid animation preset: {exc}") from exc
        return settings
