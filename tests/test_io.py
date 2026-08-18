from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splat_animator.io import detect_project, inspect_source, load_scene, read_ply_header

from .helpers import write_gaussian_ply


def test_binary_gaussian_ply_header_and_loading(tmp_path: Path) -> None:
    source = write_gaussian_ply(tmp_path / "scene.ply", count=20)
    header = read_ply_header(source)
    assert header.vertex_count == 20
    assert header.is_gaussian
    assert header.sh_degree == 0
    assert inspect_source(source).gaussian

    scene = load_scene(source, budget=7)
    assert scene.count == 7
    assert scene.original_count == 20
    assert scene.packed.shape == (7, 13)
    assert scene.packed.dtype == np.float32
    assert np.all(scene.opacity > 0.5)
    assert np.all(np.isfinite(scene.covariance_a))
    assert scene.radius > 0


def test_ascii_rgb_point_cloud_loading(tmp_path: Path) -> None:
    source = tmp_path / "points.ply"
    source.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 3",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 0 255 0 0",
                "1 0 0 0 255 0",
                "0 1 0 0 0 255",
            )
        )
        + "\n",
        encoding="ascii",
    )
    scene = load_scene(source)
    assert scene.count == 3
    assert np.allclose(scene.colors, np.eye(3, dtype=np.float32))
    assert not scene.source_info.gaussian


def test_ushort_rgb_and_alpha_are_normalized_by_declared_type(tmp_path: Path) -> None:
    source = tmp_path / "points-16-bit.ply"
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "<u2"),
            ("green", "<u2"),
            ("blue", "<u2"),
            ("alpha", "<u2"),
        ]
    )
    records = np.zeros(2, dtype=dtype)
    records["x"] = (0.0, 1.0)
    records["red"] = (32768, 65535)
    records["green"] = (65535, 0)
    records["blue"] = (0, 32768)
    records["alpha"] = (32768, 65535)
    header = "\n".join(
        (
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 2",
            "property float x",
            "property float y",
            "property float z",
            "property ushort red",
            "property ushort green",
            "property ushort blue",
            "property ushort alpha",
            "end_header",
            "",
        )
    )
    source.write_bytes(header.encode("ascii") + records.tobytes())

    scene = load_scene(source)

    midpoint = 32768 / 65535
    assert np.allclose(scene.colors, ((midpoint, 1.0, 0.0), (1.0, 0.0, midpoint)))
    assert np.allclose(scene.opacity[:, 0], (midpoint, 1.0))


def test_ascii_loader_skips_elements_declared_before_vertices(tmp_path: Path) -> None:
    source = tmp_path / "points-with-metadata.ply"
    source.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element metadata 1",
                "property int identifier",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "end_header",
                "42",
                "1 2 3",
                "4 5 6",
            )
        )
        + "\n",
        encoding="ascii",
    )

    scene = load_scene(source)

    assert scene.count == 2
    assert np.allclose(scene.positions, ((1, 2, 3), (4, 5, 6)))


def test_common_splat_loading(tmp_path: Path) -> None:
    source = tmp_path / "scene.splat"
    dtype = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    records = np.zeros(4, dtype=dtype)
    records["position"] = np.eye(4, 3, dtype=np.float32)
    records["scale"] = 0.1
    records["rgba"] = (255, 128, 0, 200)
    records["rotation"] = (255, 128, 128, 128)
    source.write_bytes(records.tobytes())
    scene = load_scene(source, budget=None)
    assert scene.count == 4
    assert scene.source_info.format_name == "SPLAT"
    assert np.allclose(scene.colors[0], (1.0, 128 / 255, 0.0))


def test_reconstrura_native_output_has_absolute_priority(tmp_path: Path) -> None:
    (tmp_path / "splat").mkdir()
    (tmp_path / "output").mkdir()
    (tmp_path / "project.json").write_text(json.dumps({"active_input_source": "frames"}))
    native = write_gaussian_ply(tmp_path / "splat" / "point_cloud.ply", count=8)
    external = write_gaussian_ply(tmp_path / "output" / "splat_999999.ply", count=9)

    detection = detect_project(tmp_path)
    assert detection.reconstrura_project
    assert detection.selected is not None
    assert detection.selected.path == native
    assert external in {candidate.path for candidate in detection.candidates}
    assert detection.suggested_output == tmp_path / "animations"
