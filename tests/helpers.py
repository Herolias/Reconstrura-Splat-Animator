from __future__ import annotations

from pathlib import Path

import numpy as np

GAUSSIAN_PROPERTIES = (
    "x",
    "y",
    "z",
    "f_dc_0",
    "f_dc_1",
    "f_dc_2",
    "opacity",
    "scale_0",
    "scale_1",
    "scale_2",
    "rot_0",
    "rot_1",
    "rot_2",
    "rot_3",
)


def write_gaussian_ply(path: Path, count: int = 128, sh_degree: int = 0) -> Path:
    if sh_degree not in {0, 1, 2, 3}:
        raise ValueError("Test Gaussian SH degree must be between 0 and 3")
    coefficient_count = (sh_degree + 1) ** 2 - 1
    rest_properties = tuple(f"f_rest_{index}" for index in range(3 * coefficient_count))
    properties = GAUSSIAN_PROPERTIES[:6] + rest_properties + GAUSSIAN_PROPERTIES[6:]
    rng = np.random.default_rng(7)
    records = np.zeros(count, dtype=[(name, "<f4") for name in properties])
    positions = rng.normal(size=(count, 3)).astype(np.float32)
    positions /= np.maximum(np.linalg.norm(positions, axis=1, keepdims=True), 1e-6)
    positions *= rng.uniform(0.25, 1.0, size=(count, 1))
    records["x"], records["y"], records["z"] = positions.T
    records["f_dc_0"] = np.linspace(-0.8, 0.8, count)
    records["f_dc_1"] = 0.1
    records["f_dc_2"] = np.linspace(0.8, -0.8, count)
    for index, name in enumerate(rest_properties):
        records[name] = (index + 1) * 0.01
    records["opacity"] = 1.2
    for name in ("scale_0", "scale_1", "scale_2"):
        records[name] = np.log(0.08)
    records["rot_0"] = 1.0
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
        *(f"property float {name}" for name in properties),
        "end_header",
        "",
    ]
    path.write_bytes("\n".join(header).encode("ascii") + records.tobytes())
    return path
