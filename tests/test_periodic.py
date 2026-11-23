import numpy as np
import pytest

from dg_wave.config import Material
from dg_wave.connectivity import pair_periodic_box
from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.physics import volume_rhs
from dg_wave.waves import plane_wave, wave_parameters


@pytest.mark.parametrize("divisions", [1, 2, 3])
def test_periodic_pairing(divisions):
    lengths = np.array([1.2, 0.9, 1.5])
    prepared = prepare_mesh(box_mesh(divisions, lengths), 2)
    connection = pair_periodic_box(
        prepared.mesh, prepared.reference, prepared.geometry, prepared.connectivity,
    )
    assert np.all(connection.neighbours >= 0)
    for element, face in np.ndindex(connection.neighbours.shape):
        other = connection.neighbours[element, face]
        rf = connection.neighbour_faces[element, face]
        permutation = connection.permutations[element, face]
        assert connection.neighbours[other, rf] == element
        difference = (
            prepared.geometry.face_nodes[element, face]
            - prepared.geometry.face_nodes[other, rf, permutation]
        ) / lengths
        np.testing.assert_allclose(difference, np.round(difference), atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(
            prepared.geometry.normals[element, face],
            -prepared.geometry.normals[other, rf], atol=1e-10, rtol=1e-10,
        )


def test_nonbox_periodic_boundary_rejected():
    box = box_mesh()
    prepared = prepare_mesh(TetraMesh(box.vertices, box.cells[:-1]), 1)
    with pytest.raises(ValueError, match="six box planes"):
        pair_periodic_box(prepared.mesh, prepared.reference, prepared.geometry, prepared.connectivity)


@pytest.mark.parametrize("kind", ["p", "s"])
def test_analytic_wave_solves_elastic_pde(kind):
    material = Material(1.0, 8 / 3, 1.0)
    points = np.array([[0.17, 0.28, 0.31], [0.52, 0.61, 0.73]])
    k, _, _, speed = wave_parameters(material, kind, (1, 1, 0))
    amplitude = plane_wave(np.zeros((1, 3)), -np.pi / (2 * speed * np.linalg.norm(k)),
                           material, kind, (1, 1, 0))[0]
    phase = points @ k - speed * np.linalg.norm(k) * 0.03
    gradient = np.cos(phase)[:, None, None] * amplitude[None, :, None] * k
    expected = -speed * np.linalg.norm(k) * np.cos(phase)[:, None] * amplitude
    np.testing.assert_allclose(volume_rhs(gradient, material), expected, atol=1e-10, rtol=1e-10)
