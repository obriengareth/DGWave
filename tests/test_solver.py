import numpy as np
import pytest

from dg_wave.config import Material
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.physics import stress_matrix, wave_speeds
from dg_wave.solver import ElasticSolver, conservative_timestep
from dg_wave.waves import plane_wave


MATERIAL = Material(1.0, 8 / 3, 1.0)


@pytest.mark.parametrize("degree", [1, 2, 3])
def test_zero_constant_and_periodic_energy(degree):
    prepared = prepare_mesh(box_mesh(), degree)
    solver = ElasticSolver(prepared, MATERIAL)
    zero = np.zeros(solver.state_shape)
    np.testing.assert_array_equal(solver.advance(zero, solver.max_timestep, 4), zero)
    constant = np.broadcast_to(np.arange(9) / 7, solver.state_shape)
    np.testing.assert_allclose(solver.rhs(constant, 0.0), 0.0, atol=1e-10, rtol=1e-10)
    state = np.random.default_rng(16).normal(size=solver.state_shape)
    initial_energy = float(solver.energy(state))
    rate = float(solver.inner_product(state, solver.rhs(state, 0.0)))
    traces = np.einsum("fij,ejc->efic", prepared.reference.trace, state)
    connection = solver.connection
    remote = traces[
        connection.neighbours[..., None], connection.neighbour_faces[..., None],
        connection.permutations,
    ]
    normal = prepared.geometry.normals[:, :, None, :]
    dv = remote[..., :3] - traces[..., :3]
    dt = np.einsum("...ij,...j->...i", stress_matrix(remote) - stress_matrix(traces), normal)
    dvn = np.sum(dv * normal, axis=-1, keepdims=True)
    dtn = np.sum(dt * normal, axis=-1, keepdims=True)
    dvt, dtt = dv - dvn * normal, dt - dtn * normal
    surface_scale = prepared.geometry.determinants[:, None] * prepared.geometry.surface_factors

    def face_norm_squared(values):
        return np.einsum(
            "efic,fij,efjc,ef->", values, prepared.reference.face_mass, values, surface_scale,
        )

    cp, cs = wave_speeds(MATERIAL)
    zp, zs = MATERIAL.density * cp, MATERIAL.density * cs
    # Each physical face occurs twice in the oriented-face sum.
    expected_rate = -0.25 * (
        zp * face_norm_squared(dvn) + zs * face_norm_squared(dvt)
        + face_norm_squared(dtn) / zp + face_norm_squared(dtt) / zs
    )
    np.testing.assert_allclose(rate, expected_rate, atol=1e-10, rtol=1e-10)
    assert rate < 0
    # Classical RK4 is not claimed to be unconditionally energy-contracting.
    # Verify a finite trajectory at the conservative timestep, allowing
    # only roundoff-sized growth; dissipative random data has strictly negative E'.
    dt = solver.max_timestep
    previous = initial_energy
    for step in range(20):
        state = solver.step(state, step * dt, dt)
        energy = float(solver.energy(state))
        assert energy <= previous + 1e-12 * initial_energy
        previous = energy
    assert previous < initial_energy


@pytest.mark.parametrize("kind", ["p", "s"])
def test_smooth_periodic_rk4_energy(kind):
    prepared = prepare_mesh(box_mesh(2), 2)
    solver = ElasticSolver(prepared, MATERIAL)
    state = plane_wave(prepared.geometry.nodes, 0.0, MATERIAL, kind)
    energy0 = float(solver.energy(state))
    rate = float(solver.inner_product(state, solver.rhs(state, 0.0)))
    assert rate <= 1e-10 * energy0
    # Check actual RK4 energies at the *full* proposed safety timestep.
    for step in range(20):
        state = solver.step(state, step * solver.max_timestep, solver.max_timestep)
        assert float(solver.energy(state)) <= energy0 * (1 + 1e-11)


def test_timestep_scales_with_geometry_degree_and_speed():
    base = prepare_mesh(box_mesh(), 1)
    refined = prepare_mesh(box_mesh(2), 1)
    higher = prepare_mesh(box_mesh(), 3)
    np.testing.assert_allclose(
        conservative_timestep(refined, MATERIAL), conservative_timestep(base, MATERIAL) / 2,
    )
    np.testing.assert_allclose(
        conservative_timestep(higher, MATERIAL), conservative_timestep(base, MATERIAL) / 4,
    )
    faster = Material(1.0, 4 * MATERIAL.bulk_modulus, 4 * MATERIAL.shear_modulus)
    np.testing.assert_allclose(
        conservative_timestep(base, faster), conservative_timestep(base, MATERIAL) / 2,
    )
    with pytest.raises(ValueError, match="conservative limit"):
        ElasticSolver(base, MATERIAL).advance(np.zeros((6, 4, 9)), 1.0, 1)
