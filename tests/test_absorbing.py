import math

import numpy as np
import pytest

from dg_wave.config import Material
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.physics import stress_matrix, wave_speeds
from dg_wave.solver import ElasticSolver
from dg_wave.waves import plane_wave


@pytest.mark.parametrize("kind", ["p", "s"])
def test_normal_packet_exit_and_reflection(kind):
    material = Material(1.0, 8 / 3, 1.0)
    prepared = prepare_mesh(box_mesh((24, 1, 1)), 2)
    solver = ElasticSolver(prepared, material, ("absorbing", "periodic", "periodic"))
    amplitude = plane_wave(np.array([[0.25, 0, 0]]), 0.0, material, kind)[0]
    envelope = np.exp(-((prepared.geometry.nodes[..., 0] - 0.3) / 0.1)**2)
    initial = envelope[..., None] * amplitude
    axis = 0 if kind == "p" else 1
    speed = wave_speeds(material)[0 if kind == "p" else 1]
    final_time = 1.1 / speed  # outgoing center at 1.4; a reflection would be centered at 0.6
    steps = math.ceil(final_time / solver.max_timestep)
    final = np.asarray(solver.advance(initial, final_time / steps, steps).block_until_ready())
    traction = np.asarray(stress_matrix(final))[..., axis, 0]
    reflected = (final[..., axis] + traction / (material.density * speed)) / 2
    mass, determinants = prepared.reference.mass, prepared.geometry.determinants

    def norm_squared(values):
        return np.einsum("ei,ij,ej,e->", values, mass, values, determinants)

    reflection = math.sqrt(norm_squared(reflected) / norm_squared(initial[..., axis]))
    remaining_energy = float(solver.energy(final) / solver.energy(initial))
    print(f"{kind.upper()} packet: reflected amplitude ratio={reflection:.6g}, remaining energy={remaining_energy:.6g}")
    assert reflection < 0.01
    assert remaining_energy < 0.01
