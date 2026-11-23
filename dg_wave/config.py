"""Small, strict TOML schema for the CPU elastic solver (SI units)."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from dg_wave.runtime import require_cpu_backend

if TYPE_CHECKING:
    import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


@dataclass(frozen=True)
class Material:
    density: float | np.ndarray
    bulk_modulus: float | np.ndarray
    shear_modulus: float | np.ndarray


@dataclass(frozen=True)
class Sponge:
    width: float
    rate: float
    faces: tuple[str, ...]


@dataclass(frozen=True)
class Attenuation:
    band: tuple[float, float]
    reference_frequency: float
    qp: float | None
    qs: float | None
    mechanisms: int
    material_q: tuple[tuple[int, float, float], ...] = ()


@dataclass(frozen=True)
class Source:
    position: tuple[float, float, float]
    frequency: float
    amplitude: float
    width: float | None = None
    delay: float | None = None
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0)
    band: tuple[float, float] | None = None
    ramp_time: float | None = None
    kind: str = "force"
    moment_angles: tuple[float, float, float] | None = None
    normalization: float | None = None
    waveform: str | None = None
    duration: float | None = None
    temporal_normalization: float | None = None


@dataclass(frozen=True)
class Config:
    mesh: Path
    material: Material | None
    degree: int
    source: Source
    dt: float
    steps: int
    backend: str
    output: Path
    output_interval: int
    boundaries: tuple[str, str, str] = ("periodic", "periodic", "periodic")
    receivers: tuple[tuple[float, float, float], ...] = ()
    materials: tuple[tuple[int, Material], ...] = ()
    free_surface: bool = False
    sponge: Sponge | None = None
    attenuation: Attenuation | None = None
    free_surface_tags: tuple[int, ...] = ()
    surface_interval: int | None = None
    receiver_components: int = 9


def _table(
    data: dict[str, Any], name: str, keys: set[str],
    optional: set[str] | frozenset[str] = frozenset(),
) -> dict[str, Any]:
    table = data.get(name)
    if not isinstance(table, dict) or not keys <= set(table) or set(table) - keys - optional:
        raise ValueError(
            f"{name} must contain exactly: {', '.join(sorted(keys))}; "
            f"optional: {', '.join(sorted(optional))}"
        )
    return table


def _number(value: Any, name: str, *, positive: bool = True) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be a {qualifier} number")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"{name} must be nonempty text without NUL characters")
    return value


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite coordinates")
    x, y, z = (_number(item, name, positive=False) for item in value)
    return x, y, z


def parse_config(data: dict[str, Any], base: Path = Path(".")) -> Config:
    sections = {
        "mesh", "discretization", "source", "time", "runtime", "output"
    }
    if not sections <= set(data) or set(data) - sections - {
        "material", "materials", "boundary", "receivers", "sponge", "attenuation",
    } or (("material" in data) == ("materials" in data)):
        raise ValueError(
            f"config must contain exactly: {', '.join(sorted(sections))}; "
            "and exactly one of 'material' or 'materials'; optional boundary/receivers/sponge"
        )
    mesh = _table(data, "mesh", {"path"})
    material = None
    materials = []
    if "material" in data:
        values = _table(data, "material", {"density", "bulk_modulus", "shear_modulus"})
        material = Material(**{
            name: _number(value, f"material.{name}") for name, value in values.items()
        })
    else:
        regions = data["materials"]
        if not isinstance(regions, dict) or not regions:
            raise ValueError("materials must be a nonempty table keyed by positive physical tags")
        for key in sorted(regions):
            if not isinstance(key, str) or not key.isdecimal() or int(key) < 1 or str(int(key)) != key:
                raise ValueError("materials keys must be canonical positive integer physical tags")
            values = _table(regions, key, {"rho", "vp", "vs"})
            rho, vp, vs = (_number(values[name], f"materials.{key}.{name}") for name in ("rho", "vp", "vs"))
            bulk = rho * (vp * vp - (4 / 3) * vs * vs)
            shear = rho * vs * vs
            if not math.isfinite(bulk) or not math.isfinite(shear) or bulk <= 0 or shear <= 0:
                raise ValueError(f"materials.{key} requires finite moduli and vp^2 > 4*vs^2/3")
            materials.append((int(key), Material(rho, bulk, shear)))
    discretization = _table(data, "discretization", {"degree"})
    source = _table(
        data, "source", {"position", "frequency", "amplitude"},
        {"width", "delay", "direction", "band", "ramp_time", "kind", "strike", "dip", "rake", "normalization",
         "waveform", "duration", "temporal_normalization"},
    )
    time = _table(data, "time", {"dt", "steps"})
    runtime = _table(data, "runtime", {"backend"})
    output = _table(data, "output", {"directory", "interval"}, {"surface_interval", "receiver_components"})
    position = _vector(source["position"], "source.position")
    source_frequency = _number(source["frequency"], "source.frequency")
    ramp_time = _number(source["ramp_time"], "source.ramp_time") if "ramp_time" in source else None
    width = _number(source["width"], "source.width") if "width" in source else None
    delay = _number(source["delay"], "source.delay", positive=False) if "delay" in source else None
    if delay is not None and delay < 0:
        raise ValueError("source.delay must be nonnegative")
    direction = _vector(source.get("direction", [1.0, 0.0, 0.0]), "source.direction")
    direction_norm = math.hypot(*direction)
    if not math.isfinite(direction_norm) or direction_norm <= 0:
        raise ValueError("source.direction must have positive finite norm")
    direction = (
        direction[0] / direction_norm, direction[1] / direction_norm,
        direction[2] / direction_norm,
    )
    source_kind = source.get("kind", "force")
    moment_angles, normalization = None, None
    waveform = source.get("waveform")
    duration = _number(source["duration"], "source.duration") if "duration" in source else None
    temporal_normalization = _number(source["temporal_normalization"], "source.temporal_normalization") if "temporal_normalization" in source else None
    if source_kind == "double_couple":
        from dg_wave.sources import double_couple_tensor

        if not {"strike", "dip", "rake", "normalization"} <= set(source) or width is None:
            raise ValueError("double_couple requires width,strike,dip,rake and global discrete normalization")
        if "direction" in source:
            raise ValueError("double_couple uses its tensor, not a force direction")
        if waveform == "band_moment_rate":
            if (
                "band" not in source or duration is None or temporal_normalization is None
                or ramp_time is None or 2*ramp_time > duration or delay is None or not 0 <= delay <= duration
            ):
                raise ValueError("band_moment_rate requires band,duration,temporal_normalization,ramp_time<=duration/2 and delay within duration")
        elif waveform is not None or any(key in source for key in ("ramp_time", "duration", "temporal_normalization")):
            raise ValueError("unknown double-couple waveform or unexpected gamma waveform parameters")
        moment_angles = tuple(_number(source[name], f"source.{name}", positive=False) for name in ("strike", "dip", "rake"))
        double_couple_tensor(*moment_angles)
        normalization = _number(source["normalization"], "source.normalization")
    elif source_kind != "force" or {"strike", "dip", "rake", "normalization", "waveform", "duration", "temporal_normalization"} & set(source):
        raise ValueError("source.kind must be force or double_couple with matching parameters")
    surface_interval = _integer(output["surface_interval"], "output.surface_interval", 1) if "surface_interval" in output else None
    receiver_components = _integer(output.get("receiver_components", 9), "output.receiver_components", 1)
    if receiver_components not in (3, 9):
        raise ValueError("output.receiver_components must be 3 or 9")
    if surface_interval is not None and discretization["degree"] != 2:
        raise ValueError("surface snapshots currently require degree 2")
    boundaries = ("periodic", "periodic", "periodic")
    free_surface = False
    free_surface_tags = ()
    if "boundary" in data:
        table = _table(data, "boundary", {"types"}, {"top", "free_tags"})
        boundary = table["types"]
        if not isinstance(boundary, list) or len(boundary) != 3 or any(
            kind not in ("periodic", "absorbing") for kind in boundary
        ):
            raise ValueError("boundary.types must be three 'periodic' or 'absorbing' axis types")
        boundaries = (boundary[0], boundary[1], boundary[2])
        if "top" in table:
            if table["top"] != "free" or boundaries[2] == "periodic":
                raise ValueError("boundary.top must be 'free' with nonperiodic z boundaries")
            free_surface = True
        if "free_tags" in table:
            tags = table["free_tags"]
            if (
                "top" in table or not isinstance(tags, list) or not tags
                or any(type(tag) is not int or tag < 1 for tag in tags)
                or len(set(tags)) != len(tags) or "periodic" in boundaries
            ):
                raise ValueError("boundary.free_tags requires distinct positive tags, nonperiodic axes and no boundary.top")
            free_surface_tags = tuple(tags)
            free_surface = True
    sponge = None
    if "sponge" in data:
        values = _table(data, "sponge", {"width", "rate", "faces"})
        faces = values["faces"]
        names = ("xmin", "xmax", "ymin", "ymax", "zmin", "zmax")
        if (
            not isinstance(faces, list) or not faces or any(face not in names for face in faces)
            or len(set(faces)) != len(faces)
        ):
            raise ValueError("sponge.faces must be distinct box-face names")
        if any(boundaries[names.index(face) // 2] == "periodic" for face in faces):
            raise ValueError("sponge.faces cannot include periodic faces")
        if free_surface and "zmax" in faces:
            raise ValueError("sponge.faces cannot include the physical free surface zmax")
        sponge = Sponge(
            _number(values["width"], "sponge.width"), _number(values["rate"], "sponge.rate"),
            tuple(faces),
        )
    receivers = ()
    if "receivers" in data:
        positions = _table(data, "receivers", {"positions"})["positions"]
        if not isinstance(positions, list):
            raise ValueError("receivers.positions must be a list of coordinate triples")
        receivers = tuple(_vector(point, "receivers.positions") for point in positions)
    attenuation = None
    if "attenuation" in data:
        values = _table(
            data, "attenuation", {"band", "reference_frequency", "mechanisms"},
            {"qp", "qs", "materials"},
        )
        band = values["band"]
        if not isinstance(band, list) or len(band) != 2:
            raise ValueError("attenuation.band must contain two positive frequencies")
        low, high = (_number(value, "attenuation.band") for value in band)
        reference = _number(values["reference_frequency"], "attenuation.reference_frequency")
        mechanisms = _integer(values["mechanisms"], "attenuation.mechanisms", 1)
        if not low < high or not low <= reference <= high:
            raise ValueError("attenuation band must increase and contain its reference frequency")
        if mechanisms > 5:
            raise ValueError("attenuation.mechanisms must lie in 1..5")
        if ("qp" in values) != ("qs" in values):
            raise ValueError("attenuation requires both global qp and qs or neither")
        qp = _number(values["qp"], "attenuation.qp") if "qp" in values else None
        qs = _number(values["qs"], "attenuation.qs") if "qs" in values else None
        material_q = []
        if "materials" in values:
            overrides = values["materials"]
            if not isinstance(overrides, dict) or not overrides:
                raise ValueError("attenuation.materials must be a nonempty table of material tags")
            configured_tags = {tag for tag, _ in materials}
            for key in sorted(overrides):
                if (
                    not isinstance(key, str) or not key.isdecimal()
                    or str(int(key)) != key or int(key) not in configured_tags
                ):
                    raise ValueError("attenuation.materials keys must match configured material tags")
                targets = _table(overrides, key, {"qp", "qs"})
                material_q.append((
                    int(key), _number(targets["qp"], f"attenuation.materials.{key}.qp"),
                    _number(targets["qs"], f"attenuation.materials.{key}.qs"),
                ))
        if qp is None and (not material_q or {tag for tag, _, _ in material_q} != {tag for tag, _ in materials}):
            raise ValueError("attenuation requires global qp/qs or targets for every material tag")
        attenuation = Attenuation((low, high), reference, qp, qs, mechanisms, tuple(material_q))
    source_band = None
    if "band" in source:
        values = source["band"]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("source.band must contain two positive increasing frequencies")
        source_band = tuple(_number(value, "source.band") for value in values)
        if not source_band[0] < source_band[1] or not source_band[0] <= source_frequency <= source_band[1]:
            raise ValueError("source.frequency must lie in an increasing source.band")
        if attenuation is not None and not (
            attenuation.band[0] <= source_band[0] and source_band[1] <= attenuation.band[1]
        ):
            raise ValueError("source.band must lie within attenuation.band")
    backend = _text(runtime["backend"], "runtime.backend")
    require_cpu_backend(backend)
    return Config(
        mesh=base / _text(mesh["path"], "mesh.path"),
        material=material,
        degree=_integer(discretization["degree"], "discretization.degree", 0),
        source=Source(
            position,
            source_frequency,
            _number(source["amplitude"], "source.amplitude", positive=False),
            width, delay, direction, source_band, ramp_time,
            source_kind, moment_angles, normalization,
            waveform, duration, temporal_normalization,
        ),
        dt=_number(time["dt"], "time.dt"),
        steps=_integer(time["steps"], "time.steps", 1),
        backend=backend,
        output=base / _text(output["directory"], "output.directory"),
        output_interval=_integer(output["interval"], "output.interval", 1),
        boundaries=boundaries,
        receivers=receivers,
        materials=tuple(materials),
        free_surface=free_surface,
        sponge=sponge,
        attenuation=attenuation,
        free_surface_tags=free_surface_tags,
        surface_interval=surface_interval, receiver_components=receiver_components,
    )


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        try:
            data = tomllib.load(stream)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"invalid TOML: {error}") from error
    return parse_config(data, path.parent)
