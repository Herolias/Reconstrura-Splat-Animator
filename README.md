# Reconstrura Splat Animator

Reconstrura Splat Animator creates turntable videos from Gaussian splats. A
scan can change the splat into a point cloud and back again.

Use the desktop app or the command-line renderer. Both can open Reconstrura
projects, standard 3DGS projects, and individual `.ply` or `.splat` files.

Reconstrura Splat Animator is an open-source companion project from the
developer behind [Reconstrura](https://reconstrura.com/). It works with most Gaussian splats and does not
require Reconstrura or a Reconstrura license. The Reconstrura logo is used with
permission; see the [trademark notice](TRADEMARKS.md).

## Features

- Gaussian splat and point-cloud rendering
- Object spin or camera orbit, clockwise or counter-clockwise
- Seamless loops with automatic duration and spin-speed adjustment
- Five transition effects: sweep, radial, wave, spiral, and dissolve
- One-way, return, and static modes
- Configurable direction, up axis, timing, easing, and edge softness
- GPU preview with mouse and keyboard camera controls
- MP4 H.264/H.265, ProRes MOV, and VP9 WebM output through FFmpeg
- Full HD, 4K, square, vertical, and custom resolutions
- Reconstrura and standard 3DGS project discovery
- Optional point limit for large scenes
- JSON presets

## Install

### Linux

Python 3.10+, FFmpeg, and a GPU driver with OpenGL 3.3 support are required.
The renderer uses EGL, so headless export is supported.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/reconstrura-splat-animator
```

With `uv`:

```bash
uv sync
uv run reconstrura-splat-animator
```

### Windows

Use 64-bit Windows 10 or 11 with Python 3.10–3.13. Install FFmpeg and make sure
`ffmpeg.exe` is available on `PATH`. Install the GPU manufacturer's current
driver; the renderer requires OpenGL 3.3 rather than the basic Windows OpenGL
driver.

In PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\reconstrura-splat-animator.exe
```

With `uv` in PowerShell:

```powershell
uv python install 3.13
uv sync --python 3.13
uv run reconstrura-splat-animator
```

## Reconstrura projects

Choose the project folder, not a file. Reconstrura Splat Animator gives the
native final Reconstrura output the highest priority:

```text
project/
  splat/
    point_cloud.ply       <- selected first
  animations/             <- suggested render folder
```

Files under `output/splat_*.ply` have lower priority because that directory can
contain exports from other tools. You can still select any detected file in the
Source section.

## Manual and headless use

Binary/ASCII PLY files with `x/y/z` are accepted. Gaussian PLY properties used
by the original 3DGS convention (`f_dc_*`, `opacity`, `scale_*`, `rot_*`) are
recognized, as are RGB point-cloud PLY files and the common 32-byte `.splat`
format.

```bash
reconstrura-splat-animator-render \
  --input scene.ply \
  --output promo.mp4 \
  --duration 12 --fps 30 \
  --spin-speed 30 --round-trip --seamless-loop
```

Use `--no-transformation --start-as point` for a point-cloud-only turntable, or
`--no-transformation --start-as splat` for a splat-only render.

The CLI will not replace an existing video unless `--force` is supplied. The
GUI asks before replacing one.

Click the preview before using the keyboard. `W`/`S` move forward and back,
`A`/`D` move left and right, and `Q`/`E` move down and up. Hold Shift to move
faster. Left drag orbits, middle or right drag pans, the mouse wheel zooms, and
double-click resets the view. Camera settings are stored in presets and used
for both the preview and export.

The rotation center can be automatic, the scene center, the orbit target, or a
custom X/Y/Z offset. Offsets use the scene radius as one unit.

The preview temporarily uses a lower resolution while you move the camera or
timeline, then redraws at full display resolution when you stop. Export always
uses the selected output resolution.

Run `reconstrura-splat-animator-render --help` for all render options. The GUI
can save its settings as JSON; use `--preset settings.json` to reproduce a
render.

## Rendering notes

The renderer sorts loaded Gaussians by camera depth and composites them
back-to-front with alpha blending. It uses the full projected 3D covariance but
currently supports DC color only; higher-order spherical harmonics are ignored.

Point size is defined at 1080p and scales with the preview and output
resolution. Videos use BT.709 limited-range color.

The point limit defaults to `No limit`. A non-zero limit applies to both the
preview and export. Set it back to `No limit` before export if you used a lower
value while working with a large scene.

## License

[MIT](LICENSE). Commercial use, modification, redistribution, and private use
of the source code are permitted with the copyright and license notice retained.
Dependencies, FFmpeg, GPU drivers, and trademarks remain governed by their own
terms. See [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing a
packaged application.

Contributions are welcome under the terms in [CONTRIBUTING.md](CONTRIBUTING.md).
