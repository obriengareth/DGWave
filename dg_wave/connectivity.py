"""Face neighbours and trace-node permutations based on shared vertex identities."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from dg_wave.mesh import TetraMesh
from dg_wave.reference import FACE_VERTICES, ReferenceOperators

if TYPE_CHECKING:
    from dg_wave.geometry import Geometry


@dataclass(frozen=True)
class Connectivity:
    # Row indices and stable IDs are separate; -1 denotes a boundary face.
    neighbours: NDArray[np.int64]
    neighbour_ids: NDArray[np.int64]
    neighbour_faces: NDArray[np.int64]
    permutations: NDArray[np.int64]


def face_node_permutation(
    local_vertices: ArrayLike, neighbour_vertices: ArrayLike,
    barycentric: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Return p such that neighbour_trace[p] is in local trace-node order."""
    local = list(local_vertices)
    neighbour = list(neighbour_vertices)
    if len(local) != 3 or len(set(local)) != 3 or set(local) != set(neighbour) or len(neighbour) != 3:
        raise ValueError("faces must contain the same three distinct vertex IDs")
    order = [local.index(vertex) for vertex in neighbour]
    mapped = barycentric[:, order]
    distances = np.max(np.abs(mapped[:, None, :] - barycentric[None, :, :]), axis=2)
    permutation = np.argmin(distances, axis=1)
    if (
        len(np.unique(permutation)) != len(barycentric)
        or np.max(distances[np.arange(len(barycentric)), permutation]) > 1.0e-12
    ):
        raise ValueError("face nodes are not a permutation-invariant triangular node set")
    return permutation


def build_connectivity(mesh: TetraMesh, ref: ReferenceOperators) -> Connectivity:
    nelements = len(mesh.cells)
    nface_nodes = len(ref.face_barycentric)
    neighbours = np.full((nelements, 4), -1, dtype=np.int64)
    neighbour_ids = np.full_like(neighbours, -1)
    neighbour_faces = np.full_like(neighbours, -1)
    permutations = np.tile(np.arange(nface_nodes), (nelements, 4, 1))
    faces: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for element, cell in enumerate(mesh.cells):
        for face, vertices in enumerate(FACE_VERTICES):
            key = tuple(sorted(cell[vertices]))
            owners = faces.setdefault(key, [])
            owners.append((element, face))
            if len(owners) > 2:
                raise ValueError(f"nonmanifold face with vertex IDs {key}")
    for owners in faces.values():
        if len(owners) == 1:
            continue
        (left, lf), (right, rf) = owners
        lv = mesh.cells[left, FACE_VERTICES[lf]]
        rv = mesh.cells[right, FACE_VERTICES[rf]]
        order = [list(lv).index(vertex) for vertex in rv]
        inversions = sum(order[i] > order[j] for i in range(3) for j in range(i + 1, 3))
        if inversions % 2 == 0:
            raise ValueError("interior faces must have opposite orientations")
        for element, face, other, other_face in (
            (left, lf, right, rf), (right, rf, left, lf),
        ):
            neighbours[element, face] = other
            neighbour_ids[element, face] = mesh.element_ids[other]
            neighbour_faces[element, face] = other_face
            permutations[element, face] = face_node_permutation(
                mesh.cells[element, FACE_VERTICES[face]],
                mesh.cells[other, FACE_VERTICES[other_face]], ref.face_barycentric,
            )
    for array in (neighbours, neighbour_ids, neighbour_faces, permutations):
        array.setflags(write=False)
    return Connectivity(neighbours, neighbour_ids, neighbour_faces, permutations)


def pair_periodic_box(
    mesh: TetraMesh, ref: ReferenceOperators, geometry: "Geometry", connection: Connectivity,
    axes: tuple[int, ...] = (0, 1, 2),
) -> Connectivity:
    """Pair every opposite boundary triangle by a box translation, not nearest centroids."""
    lower = mesh.vertices.min(axis=0)
    lengths = mesh.vertices.max(axis=0) - lower
    if np.any(lengths <= 0):
        raise ValueError("periodic box must have positive extent in all three directions")
    scaled_vertices = (mesh.vertices - lower) / lengths
    if len(set(axes)) != len(axes) or any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes):
        raise ValueError("periodic axes must be distinct members of (0, 1, 2)")
    sides: dict[tuple, list[tuple[int, int, int]]] = {}
    selected = []
    for element, face in np.argwhere(connection.neighbours < 0):
        points = scaled_vertices[mesh.cells[element, FACE_VERTICES[face]]]
        matches = [
            (axis, side) for axis in range(3) for side in (0, 1)
            if np.max(np.abs(points[:, axis] - side)) < 1.0e-12
        ]
        if len(matches) != 1:
            raise ValueError("periodic mesh boundary must lie on the six box planes")
        axis, side = matches[0]
        if axis not in axes:
            continue
        selected.append((element, face))
        projected = np.delete(points, axis, axis=1)
        key = (axis, tuple(sorted(map(tuple, np.round(projected, 12)))))
        sides.setdefault(key, []).append((int(element), int(face), side))
    arrays = [
        connection.neighbours.copy(), connection.neighbour_ids.copy(),
        connection.neighbour_faces.copy(), connection.permutations.copy(),
    ]
    neighbours, neighbour_ids, neighbour_faces, permutations = arrays
    for (axis, _), pair in sides.items():
        if len(pair) != 2 or pair[0][2] == pair[1][2]:
            raise ValueError("opposite periodic box faces must have conforming triangle subdivisions")
        (left, lf, _), (right, rf, _) = pair
        if np.max(np.abs(geometry.normals[left, lf] + geometry.normals[right, rf])) > 1e-10:
            raise ValueError("periodic face normals must be opposite")
        for element, face, other, other_face in (
            (left, lf, right, rf), (right, rf, left, lf),
        ):
            local = np.delete(scaled_vertices[mesh.cells[element, FACE_VERTICES[face]]], axis, axis=1)
            remote = np.delete(scaled_vertices[mesh.cells[other, FACE_VERTICES[other_face]]], axis, axis=1)
            distances = np.max(np.abs(remote[:, None, :] - local[None, :, :]), axis=2)
            labels = np.argmin(distances, axis=1)
            if np.max(distances[np.arange(3), labels]) > 1e-12:
                raise ValueError("periodic face vertices do not match by translation")
            neighbours[element, face] = other
            neighbour_ids[element, face] = mesh.element_ids[other]
            neighbour_faces[element, face] = other_face
            permutations[element, face] = face_node_permutation(
                np.arange(3), labels, ref.face_barycentric,
            )
    if any(neighbours[element, face] < 0 for element, face in selected):
        raise ValueError("periodic boundary pairing is incomplete")
    for array in arrays:
        array.setflags(write=False)
    return Connectivity(*arrays)
