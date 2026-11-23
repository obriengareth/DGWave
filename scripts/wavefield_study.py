"""Initial two-level CPU surface-wavefield comparison, not an acceptance study."""

import json
from pathlib import Path
import subprocess
import sys
import time

from dg_wave.runtime import configure_cpu_environment

configure_cpu_environment()

import meshio
import numpy as np

from dg_wave.config import load_config
from dg_wave.mesh import read_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.reference import FACE_VERTICES
from dg_wave.run import receiver_interpolation
from dg_wave.surface import field_comparison, free_surface_mask, refine_tetrahedra, surface_quadrature


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "output" / "surface_wavefield"
TIMES = [0., .2, .35, .5, .65, .8, 1., 1.2]


def write_mesh(path, mesh):
    rows, faces = np.nonzero(mesh.boundary_tags)
    tags = mesh.boundary_tags[rows, faces]
    meshio.write_points_cells(
        path, mesh.vertices,
        [("tetra", mesh.cells), ("triangle", mesh.cells[rows[:, None], FACE_VERTICES[faces]])],
        file_format="gmsh22", binary=False,
        cell_data={"gmsh:physical": [mesh.physical_tags, tags], "gmsh:geometrical": [mesh.physical_tags, tags]},
    )


def prepare():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coarse = read_mesh(ROOT / "volcano.msh")
    fine, parents = refine_tetrahedra(coarse)
    if len(coarse.cells) != 384 or len(fine.cells) != 3072:
        raise ValueError("this initial experiment expects the preserved 384-cell baseline and its eight-child refinement")
    write_mesh(OUTPUT / "coarse.msh", coarse)
    write_mesh(OUTPUT / "fine.msh", fine)
    np.savez(OUTPUT / "refinement.npz", parent_rows=parents, coarse_ids=coarse.element_ids, fine_ids=fine.element_ids)
    a, b = prepare_mesh(coarse, 2), prepare_mesh(fine, 2)
    masks = [free_surface_mask(p, (10,)) for p in (a, b)]
    points, weights = surface_quadrature(b, masks[1])
    rows, interpolation = receiver_interpolation(a, points)
    reconstructed = np.einsum("ri,rij->rj", interpolation, a.geometry.nodes[rows])
    geometry_error = float(np.max(np.abs(reconstructed-points)))
    if geometry_error > 1e-9:
        raise ValueError("refined sampling points left the fixed coarse-faceted geometry")
    _, coarse_area = surface_quadrature(a, masks[0])
    np.testing.assert_allclose(coarse_area.sum(), weights.sum(), rtol=1e-12)
    np.testing.assert_allclose(a.geometry.volumes.sum(), b.geometry.volumes.sum(), rtol=1e-12)
    np.savez(OUTPUT / "sampling.npz", points=points, weights=weights)
    text = (ROOT / "volcano.toml").read_text()
    text = text.replace("delay = 0.5", "delay = 0.5\nramp_time = 0.3")
    for level in ("coarse", "fine", "coarse_half_dt"):
        filename = "fine.msh" if level == "fine" else "coarse.msh"
        config = text.replace('path = "volcano.msh"', f'path = "{OUTPUT / filename}"')
        if level == "coarse_half_dt":
            config = config.replace("dt = 0.00025", "dt = 0.000125").replace("steps = 4800", "steps = 9600")
        (OUTPUT / f"{level}.toml").write_text(config)
    return {
        "coarse_elements": len(coarse.cells), "fine_elements": len(fine.cells),
        "coarse_free_faces": int(masks[0].sum()), "fine_free_faces": int(masks[1].sum()),
        "matched_points": len(points), "surface_area_m2": float(weights.sum()),
        "coordinate_matching_max_absolute_m": geometry_error,
        "geometry": "exactly inherited original affine tetrahedra/planar terrain and material interfaces; no analytic topography resampling or geometry change",
        "refinement": "one-to-eight conforming edge-midpoint tetrahedral subdivision; internal octahedron diagonal fixed within each parent",
    }


def execute(level):
    output = OUTPUT / level
    command = [
        "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core", "-n", "4",
        sys.executable, "-m", "dg_wave.wavefield", str(OUTPUT / f"{level}.toml"),
        str(OUTPUT / "sampling.npz"), str(output), "--times", *map(str, TIMES),
    ]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
    wall = time.perf_counter()-started
    (OUTPUT / f"{level}.stdout").write_text(result.stdout)
    (OUTPUT / f"{level}.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{level} failed ({result.returncode}): {result.stderr}")
    path = output / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["launcher_wall_seconds"] = wall
    metrics["launcher_wall_method"] = "external perf_counter around blocking mpiexec subprocess, includes process startup/import/setup/compile/solve/output/finalisation"
    metrics["launcher_command"] = command
    path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"completed": level, "metrics": metrics}), flush=True)
    return metrics


def analyse(geometry, metrics):
    arrays = {}
    for level in ("coarse", "fine", "coarse_half_dt"):
        with np.load(OUTPUT / level / "surface.npz", allow_pickle=False) as data:
            arrays[level] = {key: data[key].copy() for key in data.files}
    for level in ("coarse", "coarse_half_dt"):
        for key in ("comparison_points", "area_weights", "times_s"):
            np.testing.assert_array_equal(arrays[level][key], arrays["fine"][key])
    weights, times = arrays["fine"]["area_weights"], arrays["fine"]["times_s"]
    spatial = field_comparison(arrays["coarse"]["sampled_velocity"], arrays["fine"]["sampled_velocity"], weights, times)
    temporal = field_comparison(arrays["coarse_half_dt"]["sampled_velocity"], arrays["coarse"]["sampled_velocity"], weights, times)
    ratio = np.sqrt(temporal["spacetime_difference_norm_squared"]/spatial["spacetime_difference_norm_squared"])
    if ratio > .01 or temporal["spacetime_relative_l2"] > 1e-3:
        raise RuntimeError(f"temporal control failed: temporal/spatial={ratio}, relative temporal={temporal['spacetime_relative_l2']}")
    comparison = {
        "description": "preliminary two-level WAVEFIELD differences, not demonstrated convergence order or proof of resolution",
        "geometry": geometry, "spatial": spatial, "temporal_coarse": temporal,
        "temporal_over_spatial_difference_norm": float(ratio),
        "temporal_control_targets": {"temporal_over_spatial": .01, "coarse_relative_temporal": .001},
        "runs": metrics, "component_for_display": "vz in m/s; positive upward",
        "source": {"position_m": [800, 800, 700], "depth_below_summit_m": 600,
                   "type": "smooth spatial Gaussian vertical body force, not an explosion",
                   "width_m": 180, "delay_s": .5, "ramp_time_s": .3,
                   "nominal_analytic_band_hz": [.5, 3.],
                   "waveform": "64 Hann/Ricker-weighted Fourier frequencies, times C2 quintic start-up ramp; NOT strictly bandlimited"},
        "limitations": [
            "Two levels give a preliminary waveform difference, not an observed convergence order or exact error.",
            "Original coarse-faceted terrain and interfaces are deliberately held fixed; this is not geometric convergence.",
            "Coarse half-dt controls temporal contamination; no arrival-time/reflection acceptance tests or PML study launched.",
            "Degree2 fields remain coarse; finite ramp changes the spectral content beyond the nominal band.",
            "GPU work remains Pending per user; all simulations are one node/four CPU cores.",
        ],
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(comparison, indent=2))
    np.savez_compressed(
        OUTPUT / "comparison.npz", points=arrays["fine"]["comparison_points"], area_weights=weights, times_s=times,
        coarse_velocity=arrays["coarse"]["sampled_velocity"], fine_velocity=arrays["fine"]["sampled_velocity"],
        coarse_half_dt_velocity=arrays["coarse_half_dt"]["sampled_velocity"],
    )
    return comparison


def main():
    geometry = prepare()
    (OUTPUT / "geometry.json").write_text(json.dumps(geometry, indent=2))
    metrics = {level: execute(level) for level in ("coarse", "fine", "coarse_half_dt")}
    comparison = analyse(geometry, metrics)
    from dg_wave.wavefield_view import write_view
    write_view(OUTPUT, comparison)
    print(json.dumps({"comparison": comparison, "view": str(OUTPUT / "view.html")}), flush=True)


if __name__ == "__main__":
    main()
