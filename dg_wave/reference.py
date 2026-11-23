"""Nodal operators on modepy's tetrahedron with vertices (-1,-1,-1), etc."""

from dataclasses import dataclass
from functools import lru_cache

import modepy as mp
import numpy as np
from numpy.typing import NDArray


# Each outward-oriented face is opposite the vertex with the same index.
FACE_VERTICES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
FACE_VERTICES.setflags(write=False)


@dataclass(frozen=True)
class ReferenceOperators:
    degree: int
    basis: mp.Basis
    nodes: NDArray[np.float64]
    differentiation: NDArray[np.float64]
    mass: NDArray[np.float64]
    face_barycentric: NDArray[np.float64]
    face_nodes: NDArray[np.float64]
    trace: NDArray[np.float64]
    face_mass: NDArray[np.float64]
    lift: NDArray[np.float64]
    normals: NDArray[np.float64]
    surface_jacobians: NDArray[np.float64]


@lru_cache(maxsize=8, typed=True)
def reference_operators(degree: int) -> ReferenceOperators:
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    shape = mp.Simplex(3)
    space = mp.PN(3, degree)
    basis = mp.orthonormal_basis_for_space(space, shape)
    nodes = mp.edge_clustered_nodes_for_space(space, shape)
    differentiation = np.array(
        mp.differentiation_matrices(basis.functions, basis.gradients, nodes)
    )
    mass = mp.mass_matrix(basis, nodes)
    triangle = mp.Simplex(2)
    face_space = mp.PN(2, degree)
    face_basis = mp.orthonormal_basis_for_space(face_space, triangle)
    triangle_nodes = mp.edge_clustered_nodes_for_space(face_space, triangle)
    triangle_mass = mp.mass_matrix(face_basis, triangle_nodes)
    uv = (triangle_nodes.T + 1.0) / 2.0
    barycentric = np.column_stack((1.0 - uv.sum(axis=1), uv))
    vertices = mp.unit_vertices_for_shape(shape).T
    face_vertices = vertices[FACE_VERTICES]
    face_nodes = np.einsum("ni,fij->fnj", barycentric, face_vertices)
    trace = np.array([
        mp.resampling_matrix(basis.functions, points.T, nodes)
        for points in face_nodes
    ])
    area_vectors = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    lengths = np.linalg.norm(area_vectors, axis=1)
    normals = area_vectors / lengths[:, None]
    surface_jacobians = lengths / 4.0
    face_mass = surface_jacobians[:, None, None] * triangle_mass
    boundary_mass = np.concatenate([
        trace[face].T @ face_mass[face] for face in range(4)
    ], axis=1)
    lift = np.linalg.solve(mass, boundary_mass)
    arrays = (
        nodes, differentiation, mass, barycentric, face_nodes, trace,
        face_mass, lift, normals, surface_jacobians,
    )
    for array in arrays:
        array.setflags(write=False)
    return ReferenceOperators(
        degree, basis, nodes.T, differentiation, mass, barycentric, face_nodes,
        trace, face_mass, lift, normals, surface_jacobians,
    )
