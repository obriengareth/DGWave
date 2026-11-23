import json
import math

import numpy as np
import pytest

from dg_wave.attenuation import fit_relaxation
from dg_wave.config import Attenuation, Material
from dg_wave.convergence import _quadrature
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.solver import ElasticSolver


MATERIAL = Material(1, 8/3, 1)
SETTINGS = Attenuation((5, 20), 10, 60, 40, 3)


def temporal_eigenmode(model, kind, frequency):
    """Solve rho*s^2+k^2*M(s)=0 for real k, Im(s)=2*pi*f, Re(s)<0."""
    omega = 2*np.pi*frequency
    tau = model.tau
    increments = model.bulk[0]+4*model.shear[0]/3 if kind == "p" else model.shear[0]
    relaxed = (
        model.relaxed.bulk_modulus[0]+4*model.relaxed.shear_modulus[0]/3
        if kind == "p" else model.relaxed.shear_modulus[0]
    )
    rho = model.reference.density[0]
    modulus = relaxed+np.sum(increments*(1j*omega*tau)/(1+1j*omega*tau))
    k = float((omega*np.sqrt(rho/modulus)).real)
    s = 1j*omega
    for _ in range(20):
        for _ in range(30):
            modulus = relaxed+np.sum(increments*s*tau/(1+s*tau))
            derivative = np.sum(increments*tau/(1+s*tau)**2)
            delta = (rho*s*s+k*k*modulus)/(2*rho*s+k*k*derivative)
            s -= delta
            if abs(delta) < 1e-13*abs(s):
                break
        else:
            raise RuntimeError("temporal complex-modulus root did not converge")
        if abs(s.imag/omega-1) < 1e-12:
            break
        k *= omega/s.imag
    else:
        raise RuntimeError("frequency-constrained wavenumber did not converge")
    assert s.real < 0 and k > 0
    return s, k


def propagation(kind, frequency, divisions=8, half_timestep=False):
    model = fit_relaxation(MATERIAL, SETTINGS)
    s, k = temporal_eigenmode(model, kind, frequency)
    length = 2*np.pi/k
    prepared = prepare_mesh(box_mesh((divisions, 1, 1), (length, length, length)), 3)
    solver = ElasticSolver(prepared, MATERIAL, attenuation=SETTINGS)
    qnodes, weights, sampling, projection = _quadrature(3)
    vertices = prepared.mesh.vertices[prepared.mesh.cells[:, 0]]
    points = vertices[:, None]+np.einsum("eij,nj->eni", prepared.geometry.jacobians, qnodes+1)
    polarization = np.array([1, 0, 0]) if kind == "p" else np.array([0, 1, 0])
    strain = (-1j*k/s)*(np.outer(polarization, [1, 0, 0])+np.outer([1, 0, 0], polarization))/2
    trace = np.trace(strain)
    deviator = strain-trace*np.eye(3)/3
    bulk, shear = model.moduli(np.array([s]))
    stress = bulk[0, 0]*trace*np.eye(3)+2*shear[0, 0]*deviator
    indices = ((0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2))
    memory = [
        (dk*trace*np.eye(3)+2*dm*deviator)*(s*tau/(1+s*tau))
        for dk, dm, tau in zip(model.bulk[0], model.shear[0], model.tau, strict=True)
    ]
    amplitude = np.concatenate((polarization, stress[indices], *(branch[indices] for branch in memory)))
    exact = (np.exp(-1j*k*points[..., 0])[..., None]*amplitude).real
    state = np.einsum("ij,ejc->eic", projection, exact)
    samples, end = 32, 2/frequency
    count = math.ceil(end/(samples*solver.max_timestep))
    if half_timestep:
        count *= 2
    dt = end/(samples*count)
    measure = prepared.geometry.determinants[:, None]*weights
    fourier = np.exp(1j*k*points[..., 0])*measure
    coefficients, times = [], np.linspace(0, end, samples+1)
    for index, t in enumerate(times):
        evaluated = np.einsum("ij,ejc->eic", sampling, np.asarray(state)[..., :3])
        coefficients.append(np.sum((evaluated@polarization)*fourier))
        if index < samples:
            state = solver.advance(state, dt, count, float(t))
    coefficients = np.asarray(coefficients)
    phase = np.unwrap(np.angle(coefficients))

    def fitted(start):
        mask = times >= start/frequency
        decay = -np.polyfit(times[mask], np.log(np.abs(coefficients[mask])), 1)[0]
        angular = np.polyfit(times[mask], phase[mask], 1)[0]
        return float(decay), float(angular/k)

    decay, speed = fitted(0.5)
    late_decay, late_speed = fitted(0.75)
    expected_decay, expected_speed = -s.real, s.imag/k
    return {
        "frequency_hz": frequency, "divisions": divisions, "degree": 3,
        "steps": count*samples, "dt": dt,
        "measured_decay_rate": decay, "analytical_decay_rate": expected_decay,
        "measured_phase_velocity": speed, "analytical_phase_velocity": expected_speed,
        "relative_decay_error": abs(decay/expected_decay-1),
        "relative_phase_velocity_error": abs(speed/expected_speed-1),
        "fit_window_decay_difference": abs(late_decay-decay)/expected_decay,
        "fit_window_phase_difference": abs(late_speed-speed)/expected_speed,
    }


@pytest.mark.parametrize("kind", ["p", "s"])
def test_complex_modulus_decay_and_phase_over_band(kind, record_property):
    records = [propagation(kind, frequency) for frequency in (5, 10, 20)]
    half = propagation(kind, 10, half_timestep=True)
    refined = propagation(kind, 10, divisions=12)
    middle = records[1]
    temporal_decay = abs(half["measured_decay_rate"]-middle["measured_decay_rate"])/middle["analytical_decay_rate"]
    temporal_phase = abs(half["measured_phase_velocity"]-middle["measured_phase_velocity"])/middle["analytical_phase_velocity"]
    record_property("measurement", json.dumps({
        "kind": kind, "records": records, "half_timestep": half, "spatial_refinement": refined,
        "temporal_relative_decay_change": temporal_decay,
        "temporal_relative_phase_change": temporal_phase,
        "method": "periodic complex-modulus temporal eigenmode, all branch stresses initialized consistently; L2 projection; fit after 0.5 periods; end 2 periods",
        "convention": "real k and complex temporal s; exact root uses analytic continuation M(s), not weak-loss alpha*c approximation",
    }))
    assert max(record["relative_decay_error"] for record in records) < 0.01
    assert max(record["relative_phase_velocity_error"] for record in records) < 1e-3
    assert max(record["fit_window_decay_difference"] for record in records) < 0.005
    assert temporal_decay < 1e-4 and temporal_phase < 1e-6
    assert refined["relative_decay_error"] < middle["relative_decay_error"]
    assert refined["relative_phase_velocity_error"] < middle["relative_phase_velocity_error"]
