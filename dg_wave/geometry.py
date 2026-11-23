"""Affine physical geometry for positively oriented, straight-sided tetrahedra."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from dg_wave.mesh import TetraMesh
from dg_wave.reference import ReferenceOperators


@dataclass(frozen=True)
class Geometry:
    nodes: NDArray[np.float64]
    face_nodes: NDArray[np.float64]
    jacobians: NDArray[np.float64]
    inverse_jacobians: NDArray[np.float64]
    determinants: NDArray[np.float64]
    volumes: NDArray[np.float64]
    normals: NDArray[np.float64]
    surface_jacobians: NDArray[np.float64]
    surface_factors: NDArray[np.float64]


def build_geometry(mesh: TetraMesh, ref: ReferenceOperators) -> Geometry:
    vertices = mesh.vertices[mesh.cells]
    jacobians = np.transpose(vertices[:, 1:] - vertices[:, :1], (0, 2, 1)) / 2.0
    if not np.isfinite(jacobians).all():
        raise ValueError("affine Jacobians exceed the finite FP64 range")
    scale = np.max(np.abs(jacobians), axis=(1, 2))
    normalized = jacobians / np.where(scale > 0, scale, 1.0)[:, None, None]
    signed_quality = np.linalg.det(normalized)
    if np.any(signed_quality < 0):
        ids = mesh.element_ids[signed_quality < 0].tolist()
        raise ValueError(f"inverted tetrahedra with element IDs {ids}")
    degenerate = signed_quality <= 64 * np.finfo(np.float64).eps
    if np.any(degenerate):
        ids = mesh.element_ids[degenerate].tolist()
        raise ValueError(f"degenerate or numerically singular tetrahedra with element IDs {ids}")
    with np.errstate(over="ignore", under="ignore"):
        determinants = signed_quality * scale**3
    if not np.isfinite(determinants).all() or np.any(determinants <= 0):
        raise ValueError("Jacobian determinants exceed the positive finite FP64 range")
    inverse = np.linalg.inv(jacobians)
    nodes = vertices[:, :1] + np.einsum("eij,nj->eni", jacobians, ref.nodes + 1.0)
    face_nodes = vertices[:, None, :1] + np.einsum(
        "eij,fnj->efni", jacobians, ref.face_nodes + 1.0
    )
    transformed_normals = np.einsum("edk,fd->efk", inverse, ref.normals)
    # This factor multiplies reference.lift, whose face masses already include
    # the reference face area (including sqrt(3) on the slanted face).
    surface_factors = np.linalg.norm(transformed_normals, axis=2)
    normals = transformed_normals / surface_factors[:, :, None]
    surface_jacobians = (
        determinants[:, None] * surface_factors * ref.surface_jacobians[None, :]
    )
    volumes = determinants * (4.0 / 3.0)
    arrays = (
        nodes, face_nodes, jacobians, inverse, determinants, volumes,
        normals, surface_jacobians, surface_factors,
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("physical geometry exceeds the finite FP64 range")
    for array in arrays:
        array.setflags(write=False)
    return Geometry(
        nodes, face_nodes, jacobians, inverse, determinants, volumes,
        normals, surface_jacobians, surface_factors,
    )
