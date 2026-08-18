from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .models import AnimationSettings


class RenderCancelled(RuntimeError):
    pass


class FrameRenderer(Protocol):
    def render_frame(
        self,
        settings: AnimationSettings,
        seconds: float,
        *,
        sort_depth: bool = True,
    ) -> bytes: ...


def _codec_arguments(settings: AnimationSettings) -> list[str]:
    quality = str(settings.quality)
    rate_arguments = (
        ["-b:v", f"{settings.bitrate_mbps:g}M"] if settings.bitrate_mbps > 0 else ["-crf", quality]
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
    renderer: FrameRenderer,
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
