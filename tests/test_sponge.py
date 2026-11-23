import json
import math

import numpy as np

from dg_wave.config import Material, Sponge
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.run import receiver_interpolation
from dg_wave.solver import ElasticSolver


def test_oblique_sh_packet_return_with_and_without_sponge(record_property):
    material = Material(1, 8 / 3, 1)
    prepared = prepare_mesh(box_mesh((24, 6, 1), (2, 1, 1)), 2)
    bare = ElasticSolver(prepared, material, ("absorbing", "periodic", "periodic"))
    damped = ElasticSolver(
        prepared, material, ("absorbing", "periodic", "periodic"),
        sponge=Sponge(0.5, 15, ("xmax",)),
    )
    points = prepared.geometry.nodes
    kx, ky = np.arange(1, 81) * (np.pi / 4), 4 * np.pi
    weights = np.exp(-0.5 * ((kx - 4 * np.pi) / 4.5)**2)
    weights /= weights.sum()
    initial = np.zeros(bare.state_shape)
    for k, weight in zip(kx, weights, strict=True):
        phase = np.cos(k * (points[..., 0] - 0.75) + ky * points[..., 1])
        omega = math.hypot(k, ky)  # cs=1; each term is an exact outgoing SH eigenmode.
        initial[..., 2] += weight * phase
        initial[..., 7] -= weight * k / omega * phase
        initial[..., 8] -= weight * ky / omega * phase
    y = (np.arange(16) + 0.5) / 16
    rows, interpolation = receiver_interpolation(prepared, [(0.85, value, 0.5) for value in y])
    fourier = np.exp(-1j * ky * y) * (2 / len(y))
    final_time, chunk = 4.5, 32
    chunks = math.ceil(final_time / (min(bare.max_timestep, damped.max_timestep) * chunk))
    dt = final_time / (chunks * chunk)
    times = np.arange(chunks + 1) * chunk * dt
    records = []
    for solver in (bare, damped):
        state = initial
        amplitudes = []
        for index in range(chunks + 1):
            values = np.einsum("ri,ri->r", interpolation, np.asarray(state)[rows, :, 2])
            amplitudes.append(abs(values @ fourier))
            if index < chunks:
                state = solver.advance(state, dt, chunk, index * chunk * dt)
        incident = float(np.max(np.asarray(amplitudes)[times < 0.8]))
        returned = float(np.max(np.asarray(amplitudes)[times >= 2.0]))
        records.append({"incident_amplitude": incident, "returned_amplitude": returned, "ratio": returned / incident})
    record_property("measurement", json.dumps({
        "case": "positive-kx SH Fourier packet; carrier 45 degrees; x absorbing, y/z periodic",
        "domain": [2, 1, 1], "divisions": [24, 6, 1], "degree": 2,
        "center": 0.75, "monitor_x": 0.85, "ky": ky, "kx_center": 4 * np.pi,
        "kx_spectral_width": 4.5, "sponge_width": 0.5, "sponge_rate": 15,
        "returned_window": [2.0, final_time], "dt": dt, "steps": chunks * chunk,
        "without_sponge": records[0], "with_sponge": records[1],
        "method": "peak late transverse Fourier amplitude / early incident amplitude on monitor plane",
        "limitation": "case-specific numerical returned amplitude, includes packet/discretization error; not a universal bound",
    }))
    assert records[0]["ratio"] > 0.03
    assert records[1]["ratio"] < 0.5 * records[0]["ratio"]
