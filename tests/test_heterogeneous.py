import json

import numpy as np
import pytest

from dg_wave.config import Material, Sponge
from dg_wave.materials import COEFFICIENTS
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.physics import wave_speeds
from dg_wave.solver import ElasticSolver


FIRST = Material(1.0, 8 / 3, 1.0)
SECOND = Material(2.0, 14.16, 2.88)


def packet(points, material, kind, axis, center, width, direction=1):
    normal = np.eye(3)[axis] * direction
    polarization = np.eye(3)[axis if kind == "p" else (axis + 1) % 3]
    speed = wave_speeds(material)[0 if kind == "p" else 1]
    mu, lam = material.shear_modulus, material.bulk_modulus - 2 * material.shear_modulus / 3
    stress = -(lam * np.dot(polarization, normal) * np.eye(3) + mu * (
        np.outer(polarization, normal) + np.outer(normal, polarization)
    )) / speed
    state = np.r_[polarization, stress[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]]
    return np.exp(-((points[..., axis] - center) / width)**2)[..., None] * state


def scalar_inner(prepared, left, right, elements):
    return np.einsum(
        "ei,ij,ej,e->", left[elements], prepared.reference.mass, right[elements],
        prepared.geometry.determinants[elements],
    )


@pytest.mark.parametrize("kind", ["p", "s"])
def test_propagated_layer_reflection_and_transmission(kind, record_property):
    prepared = prepare_mesh(box_mesh((24, 1, 1), (2, 1, 1)), 3)
    lower = prepared.mesh.vertices[prepared.mesh.cells, 0].mean(axis=1) < 1
    material = Material(*(np.where(lower, getattr(FIRST, name), getattr(SECOND, name)) for name in COEFFICIENTS))
    solver = ElasticSolver(prepared, material, ("absorbing", "periodic", "periodic"))
    points = prepared.geometry.nodes
    state = packet(points, FIRST, kind, 0, 0.4, 0.09)
    cm = wave_speeds(FIRST)[0 if kind == "p" else 1]
    cp = wave_speeds(SECOND)[0 if kind == "p" else 1]
    zm, zp = FIRST.density * cm, SECOND.density * cp
    reflection, transmission = (zm - zp) / (zm + zp), 2 * zm / (zm + zp)
    final_time = 0.9 / cm
    steps = int(np.ceil(final_time / solver.max_timestep))
    actual = np.asarray(solver.advance(state, final_time / steps, steps))
    component = 0 if kind == "p" else 1
    reflected = packet(points, FIRST, kind, 0, 0.7, 0.09, -1)[..., component]
    incident = packet(points, FIRST, kind, 0, 1.3, 0.09)[..., component]
    transmitted = packet(points, SECOND, kind, 0, 1 + 0.3 * cp / cm, 0.09 * cp / cm)[..., component]
    measured_r = scalar_inner(prepared, reflected, actual[..., component] - incident, lower) / scalar_inner(prepared, reflected, reflected, lower)
    measured_t = scalar_inner(prepared, transmitted, actual[..., component], ~lower) / scalar_inner(prepared, transmitted, transmitted, ~lower)
    record_property("measurement", json.dumps({
        "kind": kind, "elements": len(prepared.mesh.cells), "degree": 3,
        "reflection": measured_r, "transmission": measured_t,
        "analytical_reflection": reflection, "analytical_transmission": transmission,
    }))
    np.testing.assert_allclose([measured_r, measured_t], [reflection, transmission], atol=0.005, rtol=0)


def test_heterogeneous_energy_and_element_constant_sponge_dissipation():
    prepared = prepare_mesh(box_mesh(2), 2)
    upper = prepared.mesh.vertices[prepared.mesh.cells, 2].mean(axis=1) >= 0.5
    material = Material(*(np.where(upper, getattr(FIRST, name), getattr(SECOND, name)) for name in COEFFICIENTS))
    base = ElasticSolver(prepared, material, ("absorbing", "periodic", "absorbing"), free_surface=True)
    damped = ElasticSolver(
        prepared, material, ("absorbing", "periodic", "absorbing"), free_surface=True,
        sponge=Sponge(0.3, 15, ("xmin", "xmax", "zmin")),
    )
    state = np.random.default_rng(16).normal(size=base.state_shape)
    damping_rate = float(base.inner_product(state, damped.rhs(state, 0) - base.rhs(state, 0)))
    expected = -float(base.inner_product(state, damped.damping[:, None, None] * state))
    np.testing.assert_allclose(damping_rate, expected, atol=1e-10, rtol=1e-10)
    assert damping_rate < 0
    assert float(base.inner_product(state, base.rhs(state, 0))) < 0
    initial = float(damped.energy(state))
    for step in range(20):
        state = damped.step(state, step * damped.max_timestep, damped.max_timestep)
        assert float(damped.energy(state)) <= initial * (1 + 1e-12)
        initial = float(damped.energy(state))


@pytest.mark.parametrize("kind", ["p", "s"])
def test_top_free_surface_returns_normal_packet(kind, record_property):
    prepared = prepare_mesh(box_mesh((1, 1, 24)), 3)
    solver = ElasticSolver(prepared, FIRST, ("periodic", "periodic", "absorbing"), free_surface=True)
    points = prepared.geometry.nodes
    state = packet(points, FIRST, kind, 2, 0.3, 0.075)
    speed = wave_speeds(FIRST)[0 if kind == "p" else 1]
    final_time = 1.4 / speed
    steps = int(np.ceil(final_time / solver.max_timestep))
    actual = np.asarray(solver.advance(state, final_time / steps, steps))
    expected = packet(points, FIRST, kind, 2, 0.3, 0.075, -1)
    component = 2 if kind == "p" else 0
    shape = expected[..., component]
    measured = scalar_inner(prepared, shape, actual[..., component], slice(None)) / scalar_inner(prepared, shape, shape, slice(None))
    record_property("measurement", json.dumps({
        "kind": kind, "reflected_velocity_amplitude": measured,
        "analytical_reflected_velocity_amplitude": 1.0,
        "elements": len(prepared.mesh.cells), "degree": 3,
    }))
    np.testing.assert_allclose(measured, 1.0, atol=0.005, rtol=0)
    assert np.count_nonzero(solver.free_faces) == 2
