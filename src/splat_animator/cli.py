from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

from .io import detect_project, load_scene
from .models import AnimationSettings
from .renderer import GpuRenderer, expected_extension, render_video


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reconstrura-splat-animator-render",
        description="Render a rotating Gaussian splat or point-cloud video.",
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--input", type=Path, help="Input .ply or .splat file")
    source.add_argument("--project", type=Path, help="Reconstrura or standard 3DGS project folder")
    parser.add_argument("--output", type=Path, help="Output video path")
    parser.add_argument("--force", action="store_true", help="Replace an existing output video")
    parser.add_argument("--preset", type=Path, help="JSON settings file")
    parser.add_argument(
        "--inspect-project", action="store_true", help="List files found in the project"
    )
    parser.add_argument(
        "--budget", type=int, default=0, help="Maximum points to load (0 = no limit)"
    )

    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--seamless-loop",
        action="store_true",
        default=None,
        help="Adjust the duration and spin speed to create a seamless loop",
    )
    parser.add_argument("--fps", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--spin-speed", type=float, help="Degrees per second")
    parser.add_argument(
        "--direction",
        choices=("clockwise", "counter_clockwise"),
        dest="spin_direction",
    )
    parser.add_argument("--rotation-mode", choices=("camera_orbit", "object_spin"))
    parser.add_argument("--start-angle", type=float)
    parser.add_argument("--elevation", type=float)
    parser.add_argument("--distance", type=float, dest="distance_scale")
    parser.add_argument(
        "--camera-target-x",
        type=float,
        help="Orbit target X offset, measured in scene radii",
    )
    parser.add_argument(
        "--camera-target-y",
        type=float,
        help="Orbit target Y offset, measured in scene radii",
    )
    parser.add_argument(
        "--camera-target-z",
        type=float,
        help="Orbit target Z offset, measured in scene radii",
    )
    parser.add_argument(
        "--rotation-center",
        choices=("automatic", "scene", "target", "custom"),
        dest="rotation_center_mode",
        help="Rotation center (automatic uses the orbit target or scene center)",
    )
    parser.add_argument(
        "--rotation-center-x",
        type=float,
        help="Custom rotation center X offset, measured in scene radii",
    )
    parser.add_argument(
        "--rotation-center-y",
        type=float,
        help="Custom rotation center Y offset, measured in scene radii",
    )
    parser.add_argument(
        "--rotation-center-z",
        type=float,
        help="Custom rotation center Z offset, measured in scene radii",
    )
    parser.add_argument(
        "--camera-pan-x",
        type=float,
        help="Legacy camera X offset, measured in scene radii",
    )
    parser.add_argument(
        "--camera-pan-y",
        type=float,
        help="Legacy camera Y offset, measured in scene radii",
    )
    parser.add_argument("--fov", type=float)

    parser.add_argument("--start-as", choices=("splat", "point"), dest="start_representation")
    trip = parser.add_mutually_exclusive_group()
    trip.add_argument("--round-trip", action="store_true", dest="round_trip")
    trip.add_argument("--one-way", action="store_false", dest="round_trip")
    parser.set_defaults(round_trip=None)
    transformation = parser.add_mutually_exclusive_group()
    transformation.add_argument(
        "--transformation",
        action="store_true",
        dest="transformation_enabled",
        help="Enable the transition (default)",
    )
    transformation.add_argument(
        "--no-transformation",
        action="store_false",
        dest="transformation_enabled",
        help="Keep the starting splat or point-cloud view",
    )
    parser.set_defaults(transformation_enabled=None)
    parser.add_argument("--transition-start", type=float)
    parser.add_argument("--transition-duration", type=float)
    parser.add_argument("--return-hold", type=float)
    parser.add_argument("--easing", choices=("linear", "smooth", "cinematic"))
    parser.add_argument(
        "--transition-effect",
        choices=("sweep", "radial", "wave", "spiral", "dissolve"),
    )
    parser.add_argument("--scan", choices=("top_to_bottom", "bottom_to_top"), dest="scan_direction")
    parser.add_argument("--up-axis", choices=("x", "-x", "y", "-y", "z", "-z"))
    parser.add_argument("--scan-feather", type=float)

    parser.add_argument("--point-size", type=float, dest="point_radius")
    parser.add_argument("--point-opacity", type=float)
    parser.add_argument("--splat-scale", type=float)
    parser.add_argument("--splat-opacity", type=float)
    parser.add_argument("--exposure", type=float)
    parser.add_argument("--background", help="Hex color, for example #080b11")
    background = parser.add_mutually_exclusive_group()
    background.add_argument(
        "--transparent-background",
        action="store_true",
        dest="transparent_background",
        default=None,
        help="Export alpha (requires VP9)",
    )
    background.add_argument(
        "--opaque-background",
        action="store_false",
        dest="transparent_background",
        help="Render the selected background color",
    )
    alpha_mode = parser.add_mutually_exclusive_group()
    alpha_mode.add_argument(
        "--premultiplied-alpha",
        action="store_true",
        dest="premultiplied_alpha",
        default=None,
        help="Store black-composited RGB for alpha-blind players (default)",
    )
    alpha_mode.add_argument(
        "--straight-alpha",
        action="store_false",
        dest="premultiplied_alpha",
        help="Store standard straight RGB for alpha compositing",
    )
    parser.add_argument("--codec", choices=("h264", "h265", "prores", "vp9"))
    parser.add_argument("--quality", type=int, help="CRF for H.264/H.265/VP9")
    parser.add_argument(
        "--bitrate",
        "--bitrate-mbps",
        type=float,
        dest="bitrate_mbps",
        help="Target average bitrate in Mb/s (0 uses CRF)",
    )
    return parser


def _source_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.project:
        detection = detect_project(args.project)
        if args.inspect_project:
            print(f"Project: {detection.root}")
            if not detection.candidates:
                print("No .ply or .splat candidates detected")
            for index, candidate in enumerate(detection.candidates, start=1):
                detail = candidate.info.description if candidate.info else "unreadable"
                print(f"{index:>2}. {candidate.label}\n    {candidate.path}\n    {detail}")
            raise SystemExit(0)
        if not detection.selected:
            parser.error(f"No supported splat found in project: {detection.root}")
        return detection.selected.path
    if args.input:
        return args.input
    parser.error("one of --input or --project is required")


def _settings_from_args(args: argparse.Namespace) -> AnimationSettings:
    settings = AnimationSettings.from_json(args.preset) if args.preset else AnimationSettings()
    updates = {
        key: value
        for key, value in vars(args).items()
        if value is not None and key in settings.__dataclass_fields__
    }
    if args.round_trip is not None and args.transformation_enabled is None:
        updates["transformation_enabled"] = True
    if args.transparent_background is True and args.premultiplied_alpha is None:
        updates["premultiplied_alpha"] = True
    settings = replace(settings, **updates)
    settings.validate()
    return settings


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.budget < 0:
        parser.error("--budget cannot be negative")
    source = _source_from_args(args, parser)
    if not args.output:
        parser.error("--output is required for rendering")
    settings = _settings_from_args(args).resolved_for_loop()
    output = args.output
    if output.suffix.lower() != expected_extension(settings.codec):
        output = output.with_suffix(expected_extension(settings.codec))
    if output.exists() and not args.force:
        parser.error(f"output already exists (pass --force to replace it): {output}")

    if settings.seamless_loop:
        turns = abs(settings.spin_speed) * settings.duration / 360.0
        print(
            f"Seamless loop: {settings.duration:.3f}s, {settings.spin_speed:.3f}°/s, "
            f"{turns:.0f} complete turn{'s' if round(turns) != 1 else ''}, "
            f"{settings.frame_count:,} frames"
        )

    print(f"Loading {source}...")
    scene = load_scene(source, None if args.budget == 0 else args.budget)
    print(
        f"Loaded {scene.count:,} / {scene.original_count:,} Gaussians, "
        f"scene radius {scene.radius:.4g}"
    )
    started = time.monotonic()
    last_print = 0.0

    def report(done: int, total: int) -> None:
        nonlocal last_print
        now = time.monotonic()
        if done == total or now - last_print >= 0.5:
            elapsed = now - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (total - done) / rate if rate else 0.0
            print(
                f"\rRendering {done:,}/{total:,}, {remaining:,.0f}s remaining", end="", flush=True
            )
            last_print = now

    try:
        with GpuRenderer(scene) as renderer:
            print(f"Renderer: {renderer.renderer_name}")
            rendered = render_video(renderer, settings, output, progress=report)
    except KeyboardInterrupt:
        print("\nCancelled", file=sys.stderr)
        return 130
    print(f"\nSaved {rendered} in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
