from itertools import permutations

import numpy as np
import pytest

from dg_wave.connectivity import build_connectivity, face_node_permutation
from dg_wave.geometry import build_geometry
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.reference import reference_operators


TOL = 1.0e-10


@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize("order", list(permutations(range(3))))
def test_all_six_triangular_permutations(degree, order):
    barycentric = reference_operators(degree).face_barycentric
    local = np.array([17, 31, 42])
    neighbour = local[list(order)]
    permutation = face_node_permutation(local, neighbour, barycentric)
    inverse = face_node_permutation(neighbour, local, barycentric)
    vertices = np.array([[0.2, 0.1, 0.3], [1.4, -0.5, 0.6], [0.7, 0.4, 1.3]])
    local_points = barycentric @ vertices
    neighbour_points = barycentric @ vertices[list(order)]
    np.testing.assert_array_equal(np.sort(permutation), np.arange(len(barycentric)))
    np.testing.assert_array_equal(inverse[permutation], np.arange(len(barycentric)))
    np.testing.assert_allclose(local_points, neighbour_points[permutation], atol=TOL, rtol=TOL)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_box_neighbours_and_trace_nodes(degree):
    box = box_mesh()
    order = np.array([4, 0, 5, 1, 3, 2])
    mesh = TetraMesh(box.vertices, box.cells[order], box.element_ids[order])
    ref = reference_operators(degree)
    connection = build_connectivity(mesh, ref)
    geometry = build_geometry(mesh, ref)
    assert np.count_nonzero(connection.neighbours >= 0) == 12
    assert np.count_nonzero(connection.neighbours == -1) == 12
    for element, face in np.argwhere(connection.neighbours >= 0):
        other = connection.neighbours[element, face]
        other_face = connection.neighbour_faces[element, face]
        assert connection.neighbours[other, other_face] == element
        assert connection.neighbour_faces[other, other_face] == face
        assert connection.neighbour_ids[element, face] == mesh.element_ids[other]
        permutation = connection.permutations[element, face]
        inverse = connection.permutations[other, other_face]
        np.testing.assert_array_equal(inverse[permutation], np.arange(len(permutation)))
        np.testing.assert_allclose(
            geometry.face_nodes[element, face],
            geometry.face_nodes[other, other_face, permutation], atol=TOL, rtol=TOL,
        )
        np.testing.assert_allclose(
            geometry.normals[element, face], -geometry.normals[other, other_face],
            atol=TOL, rtol=TOL,
        )


def test_nonmanifold_face_rejected():
    mesh = TetraMesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 2]],
        [[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]],
    )
    with pytest.raises(ValueError, match="nonmanifold"):
        build_connectivity(mesh, reference_operators(1))


def test_same_side_interior_face_rejected():
    mesh = TetraMesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 2]],
        [[0, 1, 2, 3], [0, 1, 2, 4]],
    )
    with pytest.raises(ValueError, match="opposite orientations"):
        build_connectivity(mesh, reference_operators(1))
