"""Reproducible periodic P/S wave verification, with spatial and temporal error separated."""

import argparse
from functools import lru_cache
import json
import math

import modepy as mp
import numpy as np

from dg_wave.config import Material
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.reference import reference_operators
from dg_wave.solver import ElasticSolver
from dg_wave.waves import plane_wave, wave_parameters


@lru_cache(maxsize=3)
def _quadrature(degree):
    ref = reference_operators(degree)
    quadrature = mp.quadrature_for_space(mp.PN(3, 2 * degree + 6), mp.Simplex(3))
    sampling = mp.resampling_matrix(ref.basis.functions, quadrature.nodes, ref.nodes.T)
    projection = np.linalg.solve(ref.mass, sampling.T * quadrature.weights)
    return quadrature.nodes.T, quadrature.weights, sampling, projection


def convergence_study(degree: int, kind: str, levels=(2, 4, 8)) -> dict:
    material = Material(1.0, 8 / 3, 1.0)  # cp=2, cs=1
    k, _, polarization, speed = wave_parameters(material, kind)
    final_time = 0.1 / speed  # 0.1 wavelength of travel for either mode
    quadrature_nodes, weights, sampling, projection = _quadrature(degree)
    records = []
    for level in levels:
        prepared = prepare_mesh(box_mesh(level), degree)
        solver = ElasticSolver(prepared, material)
        vertices = prepared.mesh.vertices[prepared.mesh.cells[:, 0]]
        points = vertices[:, None, :] + np.einsum(
            "eij,nj->eni", prepared.geometry.jacobians, quadrature_nodes + 1,
        )
        exact_initial = plane_wave(points, 0.0, material, kind)
        initial = np.einsum("ij,ejc->eic", projection, exact_initial)
        steps = math.ceil(final_time / solver.max_timestep)
        dt = final_time / steps
        final = np.asarray(solver.advance(initial, dt, steps).block_until_ready())
        evaluated = np.einsum("ij,ejc->eic", sampling, final)
        exact = plane_wave(points, final_time, material, kind)
        measure = prepared.geometry.determinants[:, None] * weights

        def norm_squared(values):
            return float(np.sum(measure[..., None] * values**2))

        error_squared = norm_squared(evaluated[..., :3] - exact[..., :3])
        relative_error = math.sqrt(error_squared / norm_squared(exact[..., :3]))
        initial_evaluated = np.einsum("ij,ejc->eic", sampling, initial)
        fourier = np.exp(-1j * (points @ k)) * measure
        coefficient0 = np.sum((initial_evaluated[..., :3] @ polarization) * fourier)
        coefficient = np.sum((evaluated[..., :3] @ polarization) * fourier)
        measured_speed = float(-np.angle(coefficient / coefficient0) / (np.linalg.norm(k) * final_time))
        record = {
            "divisions": level, "elements": len(prepared.mesh.cells), "dt": dt, "steps": steps,
            "relative_velocity_l2_error": relative_error,
            "measured_speed": measured_speed, "reference_speed": speed,
            "relative_speed_error": abs(measured_speed / speed - 1),
        }
        if level == levels[-1]:
            half = np.asarray(solver.advance(initial, dt / 2, 2 * steps).block_until_ready())
            temporal_difference = np.einsum("ij,ejc->eic", sampling, (half - final)[..., :3])
            record["temporal_difference_over_spatial_error"] = math.sqrt(
                norm_squared(temporal_difference) / error_squared
            )
        records.append(record)
    errors = [record["relative_velocity_l2_error"] for record in records]
    rates = [
        math.log(errors[i] / errors[i + 1]) / math.log(levels[i + 1] / levels[i])
        for i in range(len(levels) - 1)
    ]
    return {
        "degree": degree, "kind": kind, "final_time": final_time, "records": records,
        "observed_orders": rates, "expected_asymptotic_order": degree + 1,
        "method": (
            "smooth periodic travelling wave; L2-projected initial data; physical quadrature "
            "of velocity error with strength 2*p+6; Fourier-phase wave speed; "
            "finest-mesh timestep-halving difference relative to spatial error"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, choices=(1, 2, 3))
    parser.add_argument("--kind", choices=("p", "s"))
    args = parser.parse_args()
    from dg_wave.runtime import require_single_process

    require_single_process()
    reports = [
        convergence_study(degree, kind)
        for degree in ((args.degree,) if args.degree else (1, 2, 3))
        for kind in ((args.kind,) if args.kind else ("p", "s"))
    ]
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
