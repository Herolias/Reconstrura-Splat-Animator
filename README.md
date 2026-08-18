# Reconstrura Splat Animator

Reconstrura Splat Animator creates turntable videos from Gaussian splats. A
scan can change the splat into a point cloud and back again.

Use the desktop app or the command-line renderer. Both can open Reconstrura
projects, standard 3DGS projects, and individual `.ply` or `.splat` files.

Reconstrura Splat Animator is an open-source companion project from the
developer behind [Reconstrura](https://reconstrura.com/). It works with most
Gaussian splats and does not require Reconstrura or a Reconstrura license. The
Reconstrura logo is used with permission; see the
[trademark notice](TRADEMARKS.md).

## Features

- Gaussian splat and point-cloud rendering
- View-dependent spherical harmonics through degree 3
- Object spin or camera orbit, clockwise or counter-clockwise
- Seamless loops with automatic duration and spin-speed adjustment
- Five transition effects: sweep, radial, wave, spiral, and dissolve
- One-way, return, and static modes
- Configurable direction, up axis, timing, easing, and edge softness
- GPU preview with mouse and keyboard camera controls
- MP4 H.264/H.265, ProRes MOV, and VP9 WebM output through FFmpeg
- Transparent-background VP9 WebM export
- Full HD, 4K, square, vertical, and custom resolutions
- Reconstrura and standard 3DGS project discovery
- Optional point limit for large scenes
- JSON presets

## Installation

The steps below install the application directly from this GitHub repository.
You do not need Reconstrura or an existing Python installation. The recommended
`uv` tool installs the correct Python version and keeps this application's
packages separate from the rest of your computer. A short installation option
for people who prefer an existing Python installation is included below.

You will need:

- A 64-bit Windows 10/11 computer or a modern 64-bit Linux desktop
- A graphics card and current driver with OpenGL 3.3 support
- An internet connection and about 1 GB of free disk space
- [Git](https://git-scm.com/downloads) to download the source code
- [FFmpeg](https://ffmpeg.org/download.html) to create video files
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) to install
  Python and the application's Python packages (recommended)

Run each command below by pasting it into the terminal and pressing Enter. The
first installation can take a few minutes while files are downloaded.

### Windows 10 or 11

1. Open **Settings > Windows Update**, install available updates, and restart.
   If necessary, get the current graphics driver directly from
   [NVIDIA](https://www.nvidia.com/en-us/drivers/),
   [AMD](https://www.amd.com/en/support.html), or
   [Intel](https://www.intel.com/content/www/us/en/support/detect.html). Do not
   rely on the basic `Microsoft Basic Display Adapter` driver, because the
   application requires OpenGL 3.3.

2. Open the Start menu, type `PowerShell`, and open Windows PowerShell.

3. Install Git, `uv`, and FFmpeg with Windows Package Manager. Windows may ask
   you to approve the installers:

   ```powershell
   winget install --id Git.Git -e --source winget
   winget install --id astral-sh.uv -e --source winget
   winget install --id Gyan.FFmpeg -e --source winget
   ```

   If `winget` is not recognized, install or update
   [App Installer](https://learn.microsoft.com/windows/package-manager/winget/)
   from the Microsoft Store. Close PowerShell and open it again after all three
   commands finish.

4. Check that the prerequisites are available. Each command should print a
   version number:

   ```powershell
   git --version
   uv --version
   ffmpeg -version
   ```

5. Download this repository and enter its folder:

   ```powershell
   git clone https://github.com/Herolias/Reconstrura-Splat-Animator.git
   cd Reconstrura-Splat-Animator
   ```

6. Install Python 3.13 and the application, then start it:

   ```powershell
   uv python install 3.13
   uv sync --python 3.13 --locked
   uv run reconstrura-splat-animator
   ```

### Linux (Ubuntu or Debian)

Reconstrura Splat Animator also works on other Linux distributions. The package
names and installation commands differ, but you only need Git, FFmpeg, `curl`,
OpenGL 3.3/EGL, and a working GPU driver. Use your distribution's package
manager for those prerequisites, then continue from step 3 below.

1. Open Terminal. On Ubuntu, you can usually press `Ctrl`+`Alt`+`T`.

2. Install Git, FFmpeg, `curl`, and the OpenGL/EGL system libraries:

   ```bash
   sudo apt update
   sudo apt install git ffmpeg curl libgl1 libegl1
   ```

   Enter your computer password if asked. Linux does not display dots or other
   characters while you type a password; that is normal.

   Intel and AMD graphics drivers are normally included with Ubuntu and Debian.
   Install all available system updates. If you have an NVIDIA card, use your
   distribution's driver manager to install its recommended NVIDIA driver, then
   restart the computer.

3. Install `uv` using its official installer:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Close Terminal and open it again after the installer finishes. This makes
   the new `uv` command available.

4. Check that the prerequisites are available. Each command should print a
   version number:

   ```bash
   git --version
   uv --version
   ffmpeg -version
   ```

5. Download this repository and enter its folder:

   ```bash
   git clone https://github.com/Herolias/Reconstrura-Splat-Animator.git
   cd Reconstrura-Splat-Animator
   ```

6. Install Python 3.13 and the application, then start it:

   ```bash
   uv python install 3.13
   uv sync --python 3.13 --locked
   uv run reconstrura-splat-animator
   ```

The renderer uses EGL, so Linux can also export from a headless system with a
working GPU driver.

### Alternative: use an existing Python installation

If you prefer not to use `uv`, install Python 3.10 or newer from your operating
system or [python.org](https://www.python.org/downloads/). From the cloned
repository folder, create a private virtual environment and install the app.

On Windows with Python 3.13:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\reconstrura-splat-animator.exe
```

On Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/reconstrura-splat-animator
```

On Ubuntu or Debian, install `python3-venv` with your package manager if Python
reports that it cannot create the virtual environment. Git, FFmpeg, and the GPU
driver are still required when installing this way.

### Starting the application later

You only need to install the prerequisites once. On later starts, open Terminal
or PowerShell, enter the repository folder, and run the application:

```text
cd Reconstrura-Splat-Animator
uv run reconstrura-splat-animator
```

These commands use the recommended `uv` installation. If you installed with
your existing Python instead, use the matching `.venv` launch command from the
alternative section above.

Keep the terminal window open while the application is running. To download
future updates, close the application and run these commands from the repository
folder:

```text
git pull
uv sync --python 3.13 --locked
```

With the existing-Python installation, run its `pip install -e .` command again
after `git pull` so that any changed dependencies are installed.

If a command says it cannot find `git`, `uv`, or `ffmpeg`, close all terminal
windows, open a new one, and try the version checks again. If the application
reports an OpenGL error, update the graphics driver and restart the computer.

### Your first render

When the application opens, use the **Source** section:

- For a Reconstrura or standard 3DGS project, click **Browse**, choose the
  project folder, and click **Find files**.
- For one `.ply` or `.splat` file, click **Choose file**, select it, and then
  click **Load file**.

Adjust the animation while watching the preview. In the **Output** section,
choose where to save the video and click **Render video**.

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

For a video with an alpha channel, enable **Transparent background** in the GUI;
this selects VP9 WebM output automatically. From the CLI, combine
`--transparent-background` with `--codec vp9`. H.264, H.265, and ProRes output
remain opaque in this application.

**Black fallback for alpha-blind players** switches on automatically when
**Transparent background** is selected and remains off for normal background
videos. It makes the WebM look like a normal black-background render in players
such as VLC and Dragon Player by storing premultiplied RGB. Disable it in the
GUI—or add `--straight-alpha` on the CLI—for editors and compositors that expect
standard straight alpha. `--premultiplied-alpha` can explicitly enable it when
overriding a preset.

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
back-to-front with alpha blending. It uses the full projected 3D covariance and
view-dependent spherical harmonics through degree 3.

Transparent VP9 WebM automatically enables premultiplied RGB. This fixes
alpha-blind playback on black, but an alpha-aware compositor that assumes
straight alpha can multiply it a second time and create dark edges. Disable
**Black fallback for alpha-blind players** to export straight (unassociated)
RGB for those compositors. General-purpose players that ignore the alpha plane
can then make normally faint Gaussian edge pixels look bright and fuzzy. The
setting is inactive for normal background videos.

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
