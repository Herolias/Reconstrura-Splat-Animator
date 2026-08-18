# Changelog

Notable changes to this project are documented here.

## Unreleased

- Added view-dependent spherical harmonics rendering through degree 3.
- Added transparent-background preview and VP9 WebM export.
- Added a premultiplied-alpha fallback that switches on with transparent output
  for alpha-blind video players, with a straight-alpha compositing opt-out.
- Added optional target-bitrate encoding with an estimated output file size.

## 0.1.0 - 2026-08-18

- Initial open-source release.
- Added desktop and command-line Gaussian splat animation workflows.
- Added PLY and SPLAT loading, interactive preview, JSON presets, and FFmpeg
  export to H.264, H.265, ProRes, and VP9.
