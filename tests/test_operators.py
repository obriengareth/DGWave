from itertools import product

import numpy as np
import pytest

from dg_wave.mesh import TetraMesh, box_mesh
from dg_wave.operators import differentiate, differentiate_host, prepare_mesh


TOL = 1.0e-10


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_cached_batched_jax_gradients(degree):
    box = box_mesh()
    transform = np.array([[1.2, 0.3, -0.1], [0.2, 0.9, 0.4], [0.1, 0.2, 1.1]])
    mesh = TetraMesh(box.vertices @ transform.T + [0.1, -0.2, 0.3], box.cells, box.element_ids)
    prepared = prepare_mesh(mesh, degree)
    assert prepare_mesh(mesh, degree) is prepared
    points = prepared.geometry.nodes
    for powers in product(range(degree + 1), repeat=3):
        if sum(powers) > degree:
            continue
        values = np.prod(points ** powers, axis=2)
        expected = np.zeros((*values.shape, 3))
        for axis in range(3):
            if powers[axis]:
                reduced = list(powers)
                reduced[axis] -= 1
                expected[:, :, axis] = powers[axis] * np.prod(points ** reduced, axis=2)
        host = differentiate_host(values, prepared)
        actual = differentiate(values, prepared).block_until_ready()
        assert str(actual.dtype) == "float64"
        assert {device.platform for device in actual.devices()} == {"cpu"}
        np.testing.assert_allclose(host, expected, atol=TOL, rtol=TOL)
        np.testing.assert_allclose(actual, expected, atol=TOL, rtol=TOL)
        np.testing.assert_allclose(actual, host, atol=TOL, rtol=TOL)
