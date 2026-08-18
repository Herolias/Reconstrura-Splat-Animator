# Third-party notices

Reconstrura Splat Animator's source code is released under the MIT License. The
wheel and source archive declare the following packages as external
dependencies; they are resolved and installed by the user's Python package
installer and are not relicensed by this project.

## Python runtime dependencies

The versions below are recorded in `uv.lock` for the 0.1.0 release. NumPy's
locked version varies with the Python interpreter.

| Component | Locked version(s) | Role | Upstream license metadata |
| --- | --- | --- | --- |
| [ModernGL](https://github.com/moderngl/moderngl) | 5.12.0 | OpenGL API | MIT |
| [glcontext](https://github.com/moderngl/glcontext) | 3.0.0 | Standalone OpenGL context | MIT |
| [NumPy](https://numpy.org/doc/stable/license.html) | 2.2.6, 2.4.6, or 2.5.2 | Array processing | BSD-3-Clause plus bundled permissive licenses |
| [PySide6 Essentials](https://doc.qt.io/qtforpython-6/) | 6.11.1 | Desktop GUI | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |
| Shiboken6 | 6.11.1 | PySide6 runtime support | LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only |

These licenses allow the dependencies to be used with MIT-licensed application
source, but their notices and conditions continue to apply. PySide6 wheels also
contain Qt shared libraries and third-party components. This application imports
Qt Core, GUI, and Widgets APIs. A distributor that freezes or bundles the
application must audit the exact files included by the bundler, preserve the
applicable notices, provide the materials required by the selected LGPL/GPL
terms, and ensure users can exercise the applicable relinking or replacement
rights. Qt publishes its own [licensing overview](https://doc.qt.io/qt-6/licensing.html)
and [third-party license list](https://doc.qt.io/qtforpython-6/licenses.html).

The `dev` extra is not part of the runtime application. Its direct tools,
pytest and Ruff, report MIT licenses; their locked transitive dependencies are
also permissively licensed.

## FFmpeg and codecs

The program starts an `ffmpeg` executable found on `PATH`; the Python packages
do not contain, link to, or distribute FFmpeg. FFmpeg is normally LGPL-2.1-or-
later, but builds that enable GPL components are governed by the GPL. In
particular, this application's H.264 and H.265 modes request the external
`libx264` and `libx265` encoders, which require a suitably enabled FFmpeg build.
The [FFmpeg legal page](https://ffmpeg.org/legal.html) and
[license documentation](https://ffmpeg.org/doxygen/trunk/md_LICENSE.html)
explain how build flags and external libraries change the binary's obligations.

Transparent output requests the external `libvpx-vp9` encoder. libvpx is not
copied into this repository or its Python distributions and is separately
available under a BSD-style license with patent-license terms. A distributor
that bundles an FFmpeg build containing libvpx must preserve the notices and
audit the exact bundled build.
See the upstream libvpx [license](https://chromium.googlesource.com/webm/libvpx/+/main/LICENSE)
and [patent grant](https://chromium.googlesource.com/webm/libvpx/+/main/PATENTS).

Codec patent or royalty rules are separate from copyright licenses and vary by
jurisdiction and use. Anyone distributing FFmpeg, a desktop application bundle,
or encoded media is responsible for reviewing the exact codecs and binaries
they distribute.

GPU drivers and system OpenGL implementations are supplied by the user's
operating system or hardware vendor and remain under their own terms.

See [TRADEMARKS.md](TRADEMARKS.md) for trademark attribution and the terms for
the bundled Reconstrura logo. This notice is informational and is not legal
advice.
