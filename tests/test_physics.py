import numpy as np
import pytest

from dg_wave.config import Material
from dg_wave.physics import (
    absorbing_states, free_surface_states, riemann_states, stress_matrix,
    surface_correction, wave_speeds,
)
from dg_wave.waves import plane_wave


MATERIAL = Material(1.0, 8 / 3, 1.0)  # cp=2, cs=1


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_riemann_impedances_and_normal_reversal(heterogeneous):
    other = Material(2.0, 4.0, 2.0) if heterogeneous else MATERIAL
    left = np.array([0.3, -0.2, 0.1, 1.2, -0.3, 0.5, 0.2, -0.7, 0.6])
    right = np.array([-0.4, 0.1, 0.5, -0.2, 0.4, 0.6, -0.1, 0.3, -0.2])
    normal = np.array([1.0, 0.0, 0.0])
    velocity, traction = riemann_states(left, right, normal, MATERIAL, other)
    zleft = MATERIAL.density * np.array(wave_speeds(MATERIAL))[[0, 1, 1]]
    zright = other.density * np.array(wave_speeds(other))[[0, 1, 1]]
    tl = left[[3, 6, 7]]
    tr = right[[3, 6, 7]]
    expected_v = (zleft * left[:3] + zright * right[:3] + tr - tl) / (zleft + zright)
    expected_t = (zright * tl + zleft * tr + zleft * zright * (right[:3] - left[:3])) / (zleft + zright)
    np.testing.assert_allclose(velocity, expected_v, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(traction, expected_t, atol=1e-13, rtol=1e-13)
    reverse_v, reverse_t = riemann_states(right, left, -normal, other, MATERIAL)
    np.testing.assert_allclose(reverse_v, velocity, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(reverse_t, -traction, atol=1e-13, rtol=1e-13)


def test_riemann_consistency_oblique_normal():
    state = np.arange(9, dtype=float) / 7
    normal = np.array([1.0, 2.0, 3.0]) / np.sqrt(14)
    velocity, traction = riemann_states(state, state, normal, MATERIAL, MATERIAL)
    np.testing.assert_allclose(velocity, state[:3], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(traction, stress_matrix(state) @ normal, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        surface_correction(state, state, normal, MATERIAL, MATERIAL),
        0.0, atol=1e-13, rtol=1e-13,
    )


@pytest.mark.parametrize("kind", ["p", "s"])
def test_absorbing_outgoing_normal_wave_and_reflection(kind):
    state = plane_wave(np.array([[0.25, 0, 0]]), 0.0, MATERIAL, kind)[0]
    normal = np.array([1.0, 0.0, 0.0])
    velocity, traction = absorbing_states(state, normal, MATERIAL)
    np.testing.assert_allclose(velocity, state[:3], atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(traction, stress_matrix(state) @ normal, atol=1e-13, rtol=1e-13)
    cp, cs = wave_speeds(MATERIAL)
    impedance = MATERIAL.density * np.array([cp, cs, cs])
    reflected_velocity = (velocity + traction / impedance) / 2
    np.testing.assert_allclose(reflected_velocity, 0.0, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("kind", ["p", "s"])
@pytest.mark.parametrize(("rho", "cp", "cs"), [(2.0, 3.0, 1.2), (0.5, 1.8, 0.8)])
def test_analytical_normal_interface_reflection_transmission(kind, rho, cp, cs):
    other = Material(rho, rho * (cp**2 - 4 * cs**2 / 3), rho * cs**2)
    incident = plane_wave(np.array([[0.25, 0, 0]]), 0.0, MATERIAL, kind)[0]
    velocity, traction = riemann_states(incident, np.zeros(9), np.array([1, 0, 0]), MATERIAL, other)
    component = 0 if kind == "p" else 1
    zm = MATERIAL.density * wave_speeds(MATERIAL)[component]
    zp = rho * (cp if kind == "p" else cs)
    reflection, transmission = (zm - zp) / (zm + zp), 2 * zm / (zm + zp)
    np.testing.assert_allclose(velocity[component] - incident[component], reflection, atol=1e-13)
    np.testing.assert_allclose(velocity[component], transmission, atol=1e-13)
    np.testing.assert_allclose(traction[component], -zp * transmission, atol=1e-13)
    np.testing.assert_allclose(reflection**2 + zp / zm * transmission**2, 1, atol=1e-13)


@pytest.mark.parametrize("kind", ["p", "s"])
def test_free_surface_zero_traction_and_velocity_doubling(kind):
    incident = plane_wave(np.array([[0.25, 0, 0]]), 0.0, MATERIAL, kind)[0]
    velocity, traction = free_surface_states(incident, np.array([1, 0, 0]), MATERIAL)
    np.testing.assert_array_equal(traction, 0)
    np.testing.assert_allclose(velocity, 2 * incident[:3], atol=1e-13)
    normal = np.array([1, 2, 3]) / np.sqrt(14)
    state = np.arange(9) / 7
    _, traction = free_surface_states(state, normal, MATERIAL)
    np.testing.assert_array_equal(traction, 0)
