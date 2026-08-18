from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_PLY_TYPES: dict[str, str] = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass(frozen=True)
class PlyProperty:
    name: str
    data_type: str
    is_list: bool = False


@dataclass(frozen=True)
class PlyHeader:
    path: Path
    format: str
    vertex_count: int
    vertex_properties: tuple[PlyProperty, ...]
    data_offset: int
    vertex_offset: int
    vertex_row_offset: int
    header_lines: int

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.vertex_properties)

    @property
    def is_gaussian(self) -> bool:
        names = set(self.property_names)
        return {"scale_0", "scale_1", "scale_2"}.issubset(names) and {
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        }.issubset(names)

    @property
    def sh_degree(self) -> int:
        count = sum(name.startswith("f_rest_") for name in self.property_names)
        if count <= 0:
            return 0
        return max(0, round(math.sqrt(count / 3 + 1) - 1))


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    vertex_count: int
    format_name: str
    gaussian: bool
    sh_degree: int = 0

    @property
    def description(self) -> str:
        kind = "Gaussian splat" if self.gaussian else "point cloud"
        sh = f", SH degree {self.sh_degree}" if self.gaussian else ""
        size_mb = self.path.stat().st_size / (1024 * 1024)
        return f"{self.vertex_count:,} points, {kind}{sh}, {size_mb:.1f} MB"


@dataclass(frozen=True)
class ProjectCandidate:
    path: Path
    label: str
    score: int
    info: SourceInfo | None = None


@dataclass(frozen=True)
class ProjectDetection:
    root: Path
    candidates: tuple[ProjectCandidate, ...]
    suggested_output: Path
    reconstrura_project: bool

    @property
    def selected(self) -> ProjectCandidate | None:
        return self.candidates[0] if self.candidates else None


@dataclass
class SceneData:
    """Renderer-ready arrays; covariance is stored as six symmetric elements."""

    positions: np.ndarray
    opacity: np.ndarray
    colors: np.ndarray
    covariance_a: np.ndarray
    covariance_b: np.ndarray
    center: np.ndarray
    radius: float
    source_info: SourceInfo
    original_count: int

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def packed(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.column_stack(
                (
                    self.positions,
                    self.opacity,
                    self.colors,
                    self.covariance_a,
                    self.covariance_b,
                )
            ),
            dtype=np.float32,
        )


def read_ply_header(path: str | Path) -> PlyHeader:
    source = Path(path)
    with source.open("rb") as handle:
        first = handle.readline()
        if first.strip() != b"ply":
            raise ValueError(f"Not a PLY file: {source}")

        fmt = ""
        element = ""
        count = 0
        element_counts: list[tuple[str, int, list[PlyProperty]]] = []
        properties: list[PlyProperty] = []
        header_lines = 1
        while True:
            raw = handle.readline()
            header_lines += 1
            if not raw:
                raise ValueError(f"Incomplete PLY header: {source}")
            try:
                line = raw.decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ValueError(f"Invalid PLY header encoding: {source}") from exc
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "format" and len(tokens) >= 2:
                fmt = tokens[1]
            elif tokens[0] == "element" and len(tokens) == 3:
                if element:
                    element_counts.append((element, count, properties))
                element, count, properties = tokens[1], int(tokens[2]), []
                if count < 0:
                    raise ValueError(f"PLY element '{element}' has a negative count")
            elif tokens[0] == "property" and element:
                if len(tokens) == 5 and tokens[1] == "list":
                    properties.append(PlyProperty(tokens[-1], tokens[-2], True))
                elif len(tokens) == 3:
                    properties.append(PlyProperty(tokens[2], tokens[1]))
                else:
                    raise ValueError(f"Malformed PLY property: {line}")
            elif tokens[0] == "end_header":
                if element:
                    element_counts.append((element, count, properties))
                break

        data_offset = handle.tell()

    if fmt not in {"binary_little_endian", "binary_big_endian", "ascii"}:
        raise ValueError(f"Unsupported PLY format '{fmt}' in {source}")
    vertex = next((item for item in element_counts if item[0] == "vertex"), None)
    if vertex is None:
        raise ValueError(f"PLY has no vertex element: {source}")

    vertex_offset = data_offset
    vertex_row_offset = 0
    for name, element_count, _props in element_counts:
        if name == "vertex":
            break
        vertex_row_offset += element_count
    if fmt != "ascii":
        endian = "<" if fmt == "binary_little_endian" else ">"
        for name, count, props in element_counts:
            if name == "vertex":
                break
            if any(prop.is_list for prop in props):
                raise ValueError("List properties before the vertex element are not supported")
            dtype = _dtype_for_properties(props, endian)
            vertex_offset += count * dtype.itemsize

    return PlyHeader(
        path=source,
        format=fmt,
        vertex_count=vertex[1],
        vertex_properties=tuple(vertex[2]),
        data_offset=data_offset,
        vertex_offset=vertex_offset,
        vertex_row_offset=vertex_row_offset,
        header_lines=header_lines,
    )


def _dtype_for_properties(properties: Iterable[PlyProperty], endian: str) -> np.dtype:
    fields: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for prop in properties:
        if prop.is_list:
            raise ValueError("List properties in the vertex element are not supported")
        if prop.data_type not in _PLY_TYPES:
            raise ValueError(f"Unsupported PLY property type: {prop.data_type}")
        duplicate = seen.get(prop.name, 0)
        seen[prop.name] = duplicate + 1
        name = prop.name if duplicate == 0 else f"{prop.name}_{duplicate}"
        fields.append((name, endian + _PLY_TYPES[prop.data_type]))
    return np.dtype(fields)


def inspect_source(path: str | Path) -> SourceInfo:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".ply":
        header = read_ply_header(source)
        return SourceInfo(
            source,
            header.vertex_count,
            "PLY",
            header.is_gaussian,
            header.sh_degree,
        )
    if suffix == ".splat":
        size = source.stat().st_size
        if size % 32:
            raise ValueError("Common .splat files must contain 32-byte records")
        return SourceInfo(source, size // 32, "SPLAT", True, 0)
    raise ValueError(f"Unsupported source type '{source.suffix}'. Choose a .ply or .splat file.")


def _candidate_info(path: Path) -> SourceInfo | None:
    try:
        return inspect_source(path)
    except (OSError, ValueError):
        return None


def detect_project(root: str | Path) -> ProjectDetection:
    project = Path(root).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"Project folder does not exist: {project}")

    candidates: dict[Path, tuple[int, str]] = {}

    def add(path: Path, score: int, label: str) -> None:
        if path.is_file() and path.suffix.lower() in {".ply", ".splat"}:
            previous = candidates.get(path)
            if previous is None or score > previous[0]:
                candidates[path] = (score, label)

    # Native Reconstrura output is authoritative, including over output/splat_*.ply.
    add(project / "splat" / "point_cloud.ply", 10_000_000, "Reconstrura final splat")

    for path in project.glob("point_cloud/iteration_*/point_cloud.ply"):
        match = re.search(r"iteration_(\d+)", str(path.parent))
        iteration = int(match.group(1)) if match else 0
        add(path, 80_000 + iteration, f"3DGS iteration {iteration:,}")

    add(project / "point_cloud.ply", 70_000, "Project root: point_cloud.ply")
    add(project / "output" / "splat.ply", 60_000, "Output: splat.ply")
    add(project / "output" / "splat.splat", 60_000, "Output: splat.splat")
    for path in project.glob("output/splat_*.ply"):
        match = re.search(r"splat_(\d+)", path.stem)
        iteration = int(match.group(1)) if match else 0
        add(path, 40_000 + min(iteration, 19_999), f"External output: {path.name}")

    for pattern in ("*.splat", "*.ply", "exports/*.ply", "exports/*.splat"):
        for path in project.glob(pattern):
            add(path, 10_000, f"Other: {path.relative_to(project)}")

    resolved: list[ProjectCandidate] = []
    for path, (score, label) in candidates.items():
        info = _candidate_info(path)
        if info is not None:
            gaussian_bonus = 500 if info.gaussian else 0
            resolved.append(ProjectCandidate(path, label, score + gaussian_bonus, info))
    resolved.sort(key=lambda item: (item.score, item.path.stat().st_mtime), reverse=True)

    reconstrura = (project / "project.json").is_file() and (project / "splat").is_dir()
    return ProjectDetection(
        root=project,
        candidates=tuple(resolved),
        suggested_output=project / "animations",
        reconstrura_project=reconstrura,
    )


def _sample_indices(count: int, budget: int | None) -> slice | np.ndarray:
    if not budget or budget <= 0 or budget >= count:
        # A slice keeps memmaps lazy and avoids an unnecessary count-sized
        # int64 array plus advanced-index copy for the default full load.
        return slice(None)
    # Midpoint stratification is deterministic and avoids always selecting endpoints.
    return np.floor((np.arange(budget, dtype=np.float64) + 0.5) * count / budget).astype(np.int64)


def _field(records: np.ndarray, name: str, default: float = 0.0) -> np.ndarray:
    if records.dtype.names and name in records.dtype.names:
        return np.asarray(records[name], dtype=np.float32)
    return np.full(records.shape[0], default, dtype=np.float32)


def _normalized_channel(records: np.ndarray, name: str, default: float = 0.0) -> np.ndarray:
    """Return a PLY color/alpha channel normalized to the 0 to 1 range."""
    if not records.dtype.names or name not in records.dtype.names:
        return np.full(records.shape[0], default, dtype=np.float32)
    raw = records[name]
    values = np.asarray(raw, dtype=np.float32)
    if np.issubdtype(raw.dtype, np.integer):
        maximum = float(np.iinfo(raw.dtype).max)
        if maximum > 1.0:
            values = values / maximum
    elif float(np.nanmax(values, initial=1.0)) > 1.5:
        # Float RGB PLYs commonly store byte-range values despite their type.
        values = values / 255.0
    return values


def _load_ply_records(header: PlyHeader, indices: slice | np.ndarray) -> np.ndarray:
    endian = "<" if header.format == "binary_little_endian" else ">"
    dtype = _dtype_for_properties(header.vertex_properties, endian)
    if header.format == "ascii":
        with header.path.open("rt", encoding="ascii", errors="strict") as handle:
            array = np.loadtxt(
                handle,
                skiprows=header.header_lines + header.vertex_row_offset,
                max_rows=header.vertex_count,
            )
        if array.ndim == 1:
            array = array[None, :]
        output = np.empty(array.shape[0], dtype=dtype)
        for column, name in enumerate(dtype.names or ()):
            output[name] = array[:, column]
        return output[indices]
    mapped = np.memmap(
        header.path,
        mode="r",
        offset=header.vertex_offset,
        dtype=dtype,
        shape=(header.vertex_count,),
    )
    return np.asarray(mapped[indices])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _quaternion_covariance(scales: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scales = np.clip(
        np.nan_to_num(scales, nan=1e-8, posinf=1e6, neginf=1e-8),
        1e-8,
        1e6,
    )
    quat = np.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-8)
    w, x, y, z = quat.T
    rotation = np.empty((quat.shape[0], 3, 3), dtype=np.float32)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - z * w)
    rotation[:, 0, 2] = 2 * (x * z + y * w)
    rotation[:, 1, 0] = 2 * (x * y + z * w)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - x * w)
    rotation[:, 2, 0] = 2 * (x * z - y * w)
    rotation[:, 2, 1] = 2 * (y * z + x * w)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)

    scaled = rotation * np.square(np.maximum(scales, 1e-8))[:, None, :]
    covariance = scaled @ np.transpose(rotation, (0, 2, 1))
    a = np.column_stack((covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 0, 2]))
    b = np.column_stack((covariance[:, 1, 1], covariance[:, 1, 2], covariance[:, 2, 2]))
    return np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)


def _scene_bounds(positions: np.ndarray) -> tuple[np.ndarray, float]:
    lower, upper = np.quantile(positions, (0.01, 0.99), axis=0)
    center = ((lower + upper) * 0.5).astype(np.float32)
    distance = np.linalg.norm(positions - center, axis=1)
    radius = float(max(np.quantile(distance, 0.99), 1e-4))
    return center, radius


def _load_ply(path: Path, budget: int | None) -> SceneData:
    header = read_ply_header(path)
    if header.vertex_count <= 0:
        raise ValueError("The source contains no vertex records")
    indices = _sample_indices(header.vertex_count, budget)
    records = _load_ply_records(header, indices)
    names = set(records.dtype.names or ())
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError("PLY vertex properties must include x, y, and z")

    positions = np.column_stack((_field(records, "x"), _field(records, "y"), _field(records, "z")))
    finite = np.all(np.isfinite(positions), axis=1)
    if not np.any(finite):
        raise ValueError("The source contains no finite 3D positions")
    if not np.all(finite):
        records = records[finite]
        positions = positions[finite]
    positions = np.asarray(positions, dtype=np.float32)

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh_c0 = 0.28209479177387814
        colors = 0.5 + sh_c0 * np.column_stack(
            (_field(records, "f_dc_0"), _field(records, "f_dc_1"), _field(records, "f_dc_2"))
        )
    elif {"red", "green", "blue"}.issubset(names):
        colors = np.column_stack(
            (
                _normalized_channel(records, "red"),
                _normalized_channel(records, "green"),
                _normalized_channel(records, "blue"),
            )
        )
    elif {"r", "g", "b"}.issubset(names):
        colors = np.column_stack(
            (
                _normalized_channel(records, "r"),
                _normalized_channel(records, "g"),
                _normalized_channel(records, "b"),
            )
        )
    else:
        colors = np.full((records.shape[0], 3), 0.78, dtype=np.float32)
    colors = np.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)
    colors = np.clip(colors, 0.0, 1.0).astype(np.float32)

    if "opacity" in names:
        opacity = _sigmoid(_field(records, "opacity"))
    elif "alpha" in names:
        opacity = _normalized_channel(records, "alpha", 1.0)
    else:
        opacity = np.ones(records.shape[0], dtype=np.float32)
    opacity = np.clip(np.nan_to_num(opacity, nan=0.0), 0.0, 1.0).astype(np.float32)

    has_covariance = {
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }.issubset(names)
    if has_covariance:
        log_scales = np.nan_to_num(
            np.column_stack(
                (
                    _field(records, "scale_0"),
                    _field(records, "scale_1"),
                    _field(records, "scale_2"),
                )
            ),
            nan=-14.0,
            posinf=6.0,
            neginf=-14.0,
        )
        scales = np.exp(np.clip(log_scales, -14.0, 6.0)).astype(np.float32)
        quat = np.column_stack(
            (
                _field(records, "rot_0", 1.0),
                _field(records, "rot_1"),
                _field(records, "rot_2"),
                _field(records, "rot_3"),
            )
        ).astype(np.float32)
    else:
        _, preliminary_radius = _scene_bounds(positions)
        spacing = preliminary_radius * 1.8 / max(records.shape[0], 1) ** (1.0 / 3.0)
        scales = np.full((records.shape[0], 3), max(spacing, 1e-6), dtype=np.float32)
        quat = np.zeros((records.shape[0], 4), dtype=np.float32)
        quat[:, 0] = 1.0

    covariance_a, covariance_b = _quaternion_covariance(scales, quat)
    center, radius = _scene_bounds(positions)
    info = inspect_source(path)
    return SceneData(
        positions=np.ascontiguousarray(positions),
        opacity=np.ascontiguousarray(opacity[:, None]),
        colors=np.ascontiguousarray(colors),
        covariance_a=np.ascontiguousarray(covariance_a),
        covariance_b=np.ascontiguousarray(covariance_b),
        center=center,
        radius=radius,
        source_info=info,
        original_count=header.vertex_count,
    )


def _load_splat(path: Path, budget: int | None) -> SceneData:
    source_size = path.stat().st_size
    if source_size % 32:
        raise ValueError("Common .splat files must contain 32-byte records")
    record_count = source_size // 32
    if record_count <= 0:
        raise ValueError("The source contains no splat records")
    indices = _sample_indices(record_count, budget)
    dtype = np.dtype(
        [
            ("position", "<f4", (3,)),
            ("scale", "<f4", (3,)),
            ("rgba", "u1", (4,)),
            ("rotation", "u1", (4,)),
        ]
    )
    mapped = np.memmap(path, mode="r", dtype=dtype, shape=(record_count,))
    records = np.asarray(mapped[indices])
    positions = np.asarray(records["position"], dtype=np.float32)
    finite = np.all(np.isfinite(positions), axis=1)
    if not np.all(finite):
        records = records[finite]
        positions = positions[finite]
    positions = np.ascontiguousarray(positions)
    if not positions.size:
        raise ValueError("The source contains no finite 3D positions")
    scales = np.maximum(np.asarray(records["scale"], dtype=np.float32), 1e-8)
    colors = np.asarray(records["rgba"][:, :3], dtype=np.float32) / 255.0
    opacity = np.asarray(records["rgba"][:, 3:4], dtype=np.float32) / 255.0
    quat = (np.asarray(records["rotation"], dtype=np.float32) - 128.0) / 128.0
    covariance_a, covariance_b = _quaternion_covariance(scales, quat)
    center, radius = _scene_bounds(positions)
    return SceneData(
        positions=positions,
        opacity=np.ascontiguousarray(opacity),
        colors=np.ascontiguousarray(colors),
        covariance_a=np.ascontiguousarray(covariance_a),
        covariance_b=np.ascontiguousarray(covariance_b),
        center=center,
        radius=radius,
        source_info=inspect_source(path),
        original_count=record_count,
    )


def load_scene(path: str | Path, budget: int | None = None) -> SceneData:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Splat file does not exist: {source}")
    if source.suffix.lower() == ".ply":
        return _load_ply(source, budget)
    if source.suffix.lower() == ".splat":
        return _load_splat(source, budget)
    raise ValueError(f"Unsupported source type: {source.suffix}")
