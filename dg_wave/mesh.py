"""Immutable straight-sided tetrahedral meshes; element IDs are not row indices."""

from dataclasses import dataclass
from pathlib import Path

import meshio
import numpy as np
from numpy.typing import ArrayLike, NDArray
from dg_wave.reference import FACE_VERTICES


@dataclass(frozen=True, eq=False)
class TetraMesh:
    vertices: NDArray[np.float64]
    cells: NDArray[np.int64]
    element_ids: NDArray[np.int64]
    physical_tags: NDArray[np.int64] | None
    boundary_tags: NDArray[np.int64] | None

    def __init__(
        self, vertices: ArrayLike, cells: ArrayLike, element_ids: ArrayLike | None = None,
        physical_tags: ArrayLike | None = None,
        boundary_tags: ArrayLike | None = None,
    ) -> None:
        points = np.array(vertices, dtype=np.float64, copy=True)
        raw_cells = np.asarray(cells)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("vertices must be finite with shape (nvertices, 3)")
        if (
            raw_cells.ndim != 2 or raw_cells.shape[1] != 4 or len(raw_cells) == 0
            or not np.issubdtype(raw_cells.dtype, np.integer)
        ):
            raise ValueError("cells must be nonempty integer tetrahedra with shape (nelements, 4)")
        if np.any(raw_cells < 0) or np.any(raw_cells >= len(points)):
            raise ValueError("cell vertex index out of range")
        connectivity = np.array(raw_cells, dtype=np.int64, copy=True)
        sorted_cells = np.sort(connectivity, axis=1)
        if np.any(np.diff(sorted_cells, axis=1) == 0):
            raise ValueError("a tetrahedron must have four distinct vertices")
        if len(np.unique(sorted_cells, axis=0)) != len(connectivity):
            raise ValueError("duplicate tetrahedra")
        raw_ids = np.asarray(
            np.arange(len(connectivity)) if element_ids is None else element_ids
        )
        if (
            raw_ids.shape != (len(connectivity),)
            or not np.issubdtype(raw_ids.dtype, np.integer)
            or np.any(raw_ids < 0)
            or np.any(raw_ids > np.iinfo(np.int64).max)
            or len(np.unique(raw_ids)) != len(raw_ids)
        ):
            raise ValueError("element_ids must be unique nonnegative int64 IDs")
        ids = np.array(raw_ids, dtype=np.int64, copy=True)
        for name, array in (
            ("vertices", points), ("cells", connectivity), ("element_ids", ids),
        ):
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        tags = None
        if physical_tags is not None:
            raw_tags = np.asarray(physical_tags)
            if (
                raw_tags.shape != (len(connectivity),)
                or not np.issubdtype(raw_tags.dtype, np.integer)
                or np.any(raw_tags <= 0) or np.any(raw_tags > np.iinfo(np.int64).max)
            ):
                raise ValueError("physical tags must be positive int64 values, one per tetrahedron")
            tags = np.array(raw_tags, dtype=np.int64, copy=True)
            tags.setflags(write=False)
        object.__setattr__(self, "physical_tags", tags)
        face_tags = None
        if boundary_tags is not None:
            raw = np.asarray(boundary_tags)
            if (
                raw.shape != (len(connectivity), 4) or not np.issubdtype(raw.dtype, np.integer)
                or np.any(raw < 0) or np.any(raw > np.iinfo(np.int64).max)
            ):
                raise ValueError("boundary tags must be nonnegative integers with shape (nelements, 4)")
            faces = np.sort(connectivity[:, FACE_VERTICES], axis=-1).reshape(-1, 3)
            _, inverse, counts = np.unique(faces, axis=0, return_inverse=True, return_counts=True)
            if np.any((raw.ravel() > 0) & (counts[inverse] != 1)):
                raise ValueError("boundary tags may select exterior faces only")
            face_tags = np.array(raw, dtype=np.int64, copy=True)
            face_tags.setflags(write=False)
        object.__setattr__(self, "boundary_tags", face_tags)

    def __reduce__(self):
        # MPI metadata serialization must restore the immutable-array invariant.
        return type(self), (self.vertices, self.cells, self.element_ids, self.physical_tags, self.boundary_tags)


def box_mesh(
    divisions: int | tuple[int, int, int] = 1, lengths: ArrayLike = (1.0, 1.0, 1.0),
) -> TetraMesh:
    """Conforming Cartesian box, six tetrahedra per cube, stable IDs from 100."""
    lengths = np.asarray(lengths, dtype=np.float64)
    counts = np.asarray((divisions,) * 3 if type(divisions) is int else divisions)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 1):
        raise ValueError("box divisions must be a positive integer or integer triple")
    if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("box lengths must be three positive finite numbers")
    corners = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ])
    tetrahedra = np.array([
            [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6],
            [0, 7, 4, 6], [0, 4, 5, 6], [0, 5, 1, 6],
    ])
    if np.all(counts == 1):
        return TetraMesh(corners * lengths, tetrahedra, np.arange(100, 106))
    nx, ny, nz = map(int, counts)
    grid = np.indices((nx + 1, ny + 1, nz + 1)).reshape(3, -1).T
    vertices = grid * (lengths / counts)
    cells = []
    for origin in np.indices((nx, ny, nz)).reshape(3, -1).T:
        indices = corners + origin
        indices = (indices[:, 0] * (ny + 1) + indices[:, 1]) * (nz + 1) + indices[:, 2]
        cells.extend(indices[tetrahedra])
    return TetraMesh(vertices, cells, np.arange(100, 100 + len(cells)))


def read_mesh(path: str | Path) -> TetraMesh:
    """Read linear tetrahedra; preserve element_id cell data when supplied.

    Otherwise assign stable IDs in imported tetrahedron order. Physical triangle
    tags select exterior faces; unsupported volume/high-order cells are rejected.
    """
    try:
        mesh = meshio.read(path)
    except (meshio.ReadError, SystemExit) as error:
        # meshio's format dispatcher can exit instead of raising its reader error.
        raise ValueError(f"meshio could not read mesh {path}: {error}") from error
    if any(block.type not in {"tetra", "triangle", "line", "vertex"} for block in mesh.cells):
        raise ValueError("only linear tetrahedra and lower-dimensional boundary cells are supported")
    indices = [index for index, block in enumerate(mesh.cells) if block.type == "tetra"]
    if not indices:
        raise ValueError("mesh contains no tetrahedra")
    cells = np.concatenate([mesh.cells[index].data for index in indices])
    ids = None
    if "element_id" in mesh.cell_data:
        ids = np.concatenate([mesh.cell_data["element_id"][index] for index in indices])
    tags = None
    for key in ("gmsh:physical", "material_tag"):
        if key in mesh.cell_data:
            tags = np.concatenate([mesh.cell_data[key][index] for index in indices])
            break
    boundary_tags = np.zeros((len(cells), 4), dtype=np.int64)
    face_rows = {}
    for element, face in np.ndindex(boundary_tags.shape):
        key = tuple(sorted(cells[element, FACE_VERTICES[face]]))
        face_rows.setdefault(key, []).append((element, face))
    seen = set()
    for index, block in enumerate(mesh.cells):
        if block.type != "triangle" or "gmsh:physical" not in mesh.cell_data:
            continue
        for triangle, tag in zip(block.data, mesh.cell_data["gmsh:physical"][index], strict=True):
            key = tuple(sorted(triangle))
            if key in seen or len(face_rows.get(key, [])) != 1 or tag <= 0:
                raise ValueError("physical boundary triangles must be unique tagged exterior tetrahedral faces")
            seen.add(key)
            boundary_tags[face_rows[key][0]] = tag
    return TetraMesh(mesh.points, cells, ids, tags, boundary_tags if seen else None)
