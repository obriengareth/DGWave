import numpy as np
import pytest

from dg_wave.geometry import build_geometry
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.reference import FACE_VERTICES, reference_operators


TOL = 1.0e-10


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_affine_maps_volumes_and_faces(degree):
    box = box_mesh()
    transform = np.array([[1.4, 0.2, 0.1], [0.1, 1.1, 0.3], [0.0, 0.2, 0.9]])
    mesh = TetraMesh(box.vertices @ transform.T + [0.2, -0.3, 1.4], box.cells, box.element_ids)
    ref = reference_operators(degree)
    geometry = build_geometry(mesh, ref)
    np.testing.assert_allclose(geometry.volumes, np.linalg.det(transform) / 6, atol=TOL, rtol=TOL)
    assert np.all(geometry.volumes > 0)
    np.testing.assert_allclose(
        geometry.jacobians @ geometry.inverse_jacobians,
        np.broadcast_to(np.eye(3), (6, 3, 3)), atol=TOL, rtol=TOL,
    )
    ref_vertices = np.array([[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
    mapped_vertices = mesh.vertices[mesh.cells[:, :1]] + np.einsum(
        "eij,nj->eni", geometry.jacobians, ref_vertices + 1.0,
    )
    np.testing.assert_allclose(mapped_vertices, mesh.vertices[mesh.cells], atol=TOL, rtol=TOL)
    np.testing.assert_allclose(
        np.einsum("fij,ejk->efik", ref.trace, geometry.nodes),
        geometry.face_nodes, atol=TOL, rtol=TOL,
    )
    faces = {}
    for element, cell in enumerate(mesh.cells):
        center = mesh.vertices[cell].mean(axis=0)
        for face, face_indices in enumerate(FACE_VERTICES):
            points = mesh.vertices[cell[face_indices]]
            vector = np.cross(points[1] - points[0], points[2] - points[0])
            np.testing.assert_allclose(
                geometry.normals[element, face], vector / np.linalg.norm(vector),
                atol=TOL, rtol=TOL,
            )
            np.testing.assert_allclose(
                geometry.surface_jacobians[element, face], np.linalg.norm(vector) / 4,
                atol=TOL, rtol=TOL,
            )
            assert np.dot(geometry.normals[element, face], points.mean(axis=0) - center) > 0
            faces.setdefault(tuple(sorted(cell[face_indices])), []).append((element, face))
    interior = [pair for pair in faces.values() if len(pair) == 2]
    assert len(interior) == 6
    assert sum(len(pair) == 1 for pair in faces.values()) == 12
    for (left, lf), (right, rf) in interior:
        np.testing.assert_allclose(
            geometry.normals[left, lf], -geometry.normals[right, rf], atol=TOL, rtol=TOL,
        )
    np.testing.assert_allclose(
        np.sum(geometry.normals * geometry.surface_jacobians[:, :, None], axis=1),
        0.0, atol=TOL, rtol=TOL,
    )
    np.testing.assert_allclose(
        geometry.surface_jacobians,
        geometry.determinants[:, None] * geometry.surface_factors * ref.surface_jacobians,
        atol=TOL, rtol=TOL,
    )
    for element in range(len(mesh.cells)):
        for axis in range(3):
            derivative = np.einsum(
                "d,dij->ij", geometry.inverse_jacobians[element, :, axis],
                ref.differentiation,
            )
            boundary = sum(
                geometry.normals[element, face, axis]
                * geometry.surface_jacobians[element, face] / ref.surface_jacobians[face]
                * ref.trace[face].T @ ref.face_mass[face] @ ref.trace[face]
                for face in range(4)
            )
            np.testing.assert_allclose(
                geometry.determinants[element]
                * (ref.mass @ derivative + derivative.T @ ref.mass),
                boundary, atol=TOL, rtol=TOL,
            )


@pytest.mark.parametrize(("height", "message"), [
    (-1.0, "inverted"), (0.0, "degenerate"), (1.0e-16, "degenerate"),
])
def test_invalid_cell_geometry(height, message):
    mesh = TetraMesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, height]], [[0, 1, 2, 3]], [77],
    )
    with pytest.raises(ValueError, match=rf"{message}.*77"):
        build_geometry(mesh, reference_operators(1))


@pytest.mark.parametrize("scale", [1.0e-8, 1.0, 1.0e8])
def test_nondegeneracy_is_scale_invariant(scale):
    box = box_mesh()
    mesh = TetraMesh(box.vertices * scale, box.cells)
    geometry = build_geometry(mesh, reference_operators(1))
    np.testing.assert_allclose(geometry.volumes / scale**3, 1 / 6, atol=TOL, rtol=TOL)
