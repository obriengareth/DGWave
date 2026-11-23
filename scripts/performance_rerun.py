"""Reproduce the archived3072-cell wavefield after the approved bounded changes."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from dg_wave.runtime import configure_cpu_environment

configure_cpu_environment()

import numpy as np

from dg_wave.config import load_config
from dg_wave.surface import field_comparison
from scripts import wavefield_next
from scripts.wavefield_study import ROOT, TIMES

OUTPUT = ROOT / "output" / "performance_rerun"
ORIGINAL = ROOT / "output" / "surface_wavefield"


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(ORIGINAL.iterdir()) if p.is_file()} | {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ("coarse", "fine", "coarse_half_dt", "veryfine", "fine_time_control", "veryfine_pilot")
        for p in sorted((ORIGINAL / name).glob("*")) if p.is_file()
    }


def scaled_error(actual, reference):
    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)
    error = float(np.max(np.abs(actual-reference)/np.maximum(1, np.abs(reference))))
    if error > 1e-10:
        raise ValueError(f"unchanged FP64 scaled tolerance exceeded: {error}")
    return error


def main():
    with zipfile.ZipFile(OUTPUT / "baseline3072.zip") as archive:
        baseline_manifest = json.loads(archive.read("sha256.json"))
    for path, digest in baseline_manifest.items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"sealed pre-change baseline changed: {path}")
    original_hashes = hashes()
    config = load_config(ORIGINAL / "fine.toml")
    if config.degree != 2 or config.dt != .00025 or config.steps != 4800:
        raise ValueError("the requested original degree2/dt.00025/4800-step configuration changed")
    output = OUTPUT / "optimized3072"
    if output.exists():
        raise ValueError("refusing to overwrite an existing performance rerun")
    (OUTPUT / "optimized3072.toml").write_bytes((ORIGINAL / "fine.toml").read_bytes())
    (OUTPUT / "sampling.npz").write_bytes((ORIGINAL / "sampling.npz").read_bytes())
    (OUTPUT / "preserved-wavefield-sha256.json").write_text(json.dumps(original_hashes, indent=2))
    binding_code = """
import json,os
from dg_wave.runtime import initialize_mpi_cpu,physical_binding
comm,_,local_rank=initialize_mpi_cpu()
record=physical_binding(comm,local_rank)
masks=comm.allgather(record['logical_cpu_mask'])
if masks!=[[0],[2],[4],[6]]:raise RuntimeError('required logical CPU0/2/4/6 binding differs from current mapping')
record['xla_flags']=os.environ.get('XLA_FLAGS','unset; installed JAX CPU defaults')
records=comm.gather(record)
if comm.rank==0:print(json.dumps(records))
"""
    launcher = ["mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core", "-n", "4", sys.executable]
    preflight = subprocess.run(
        [*launcher, "-c", binding_code], capture_output=True, text=True, check=True, timeout=40,
    )
    (OUTPUT / "pinning-preflight.json").write_text(json.dumps(json.loads(preflight.stdout), indent=2))
    wavefield_next.OUTPUT = OUTPUT
    metrics = wavefield_next.execute(
        "optimized3072", TIMES, 1200, 1300, expected_cells=3072,
    )
    if metrics["intentional_final_volume_gathers"] != 1 or metrics["surface_snapshots"] != 8:
        raise ValueError("surface-only snapshots must perform exactly one intentional final volume gather")
    for rank in metrics["ranks"]:
        if rank["logical_cpu_mask"] != [[0], [2], [4], [6]][rank["rank"]]:
            raise ValueError("actual solve affinity differs from required current physical-core preflight")
    before = json.loads((ORIGINAL / "fine" / "metrics.json").read_text())
    with np.load(ORIGINAL / "fine" / "final.npz") as reference, np.load(output / "final.npz") as actual:
        np.testing.assert_array_equal(actual["element_ids"], reference["element_ids"])
        full = scaled_error(actual["state"], reference["state"])
        physical = scaled_error(actual["state"][..., :9], reference["state"][..., :9])
        memory = scaled_error(actual["state"][..., 9:], reference["state"][..., 9:])
    with np.load(ORIGINAL / "fine" / "surface.npz") as reference, np.load(output / "surface.npz") as actual:
        for key in ("times_s", "face_positions", "triangles", "comparison_points", "area_weights",
                    "face_element_ids", "face_indices"):
            np.testing.assert_array_equal(actual[key], reference[key])
        surface = scaled_error(actual["surface_state"], reference["surface_state"])
        velocity = scaled_error(actual["sampled_velocity"], reference["sampled_velocity"])
        baseline_difference = field_comparison(
            actual["sampled_velocity"], reference["sampled_velocity"], reference["area_weights"], reference["times_s"],
        )
        with np.load(ORIGINAL / "coarse" / "surface.npz") as coarse:
            prior_metric = field_comparison(
                coarse["sampled_velocity"], reference["sampled_velocity"], reference["area_weights"], reference["times_s"],
            )
            revised_metric = field_comparison(
                coarse["sampled_velocity"], actual["sampled_velocity"], actual["area_weights"], actual["times_s"],
            )
        np.testing.assert_allclose(
            revised_metric["spacetime_relative_l2"], prior_metric["spacetime_relative_l2"], atol=1e-12, rtol=1e-10,
        )
    if hashes() != original_hashes:
        raise ValueError("an original wavefield artifact changed during the performance rerun")
    if hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest() != baseline_manifest["requirements.lock"]:
        raise ValueError("CPU environment lock changed")
    result = {
        "implemented": ["selected free-surface nodal snapshot extraction/gather",
                        "128-element bounded frequency response/Q/resolution reporting",
                        "remote-only physical9 host/MPI exchange; local mappings/permutations in shared JAX residual"],
        "problem": {"elements": 3072, "degree": 2, "mpi_processes": 4,
                    "nodes": 30720, "state_components": 27, "state_unknowns": 829440,
                    "dt_s": .00025, "steps": 4800, "duration_s": 1.2,
                    "matched_points": 1536, "saved_times": 8},
        "unchanged_fp64_tolerance": 1e-10,
        "max_scaled_errors": {"full27": full, "physical9": physical, "Maxwell18": memory,
                              "surface27": surface, "matched_velocity3": velocity},
        "new_vs_archived_matched_surface_vector_norm": baseline_difference,
        "prior_coarse_vs3072_metric": prior_metric, "coarse_vs_optimized3072_metric": revised_metric,
        "before": before, "after": metrics,
        "measured_solve_speedup": before["solve_seconds_max_rank"]/metrics["solve_seconds_max_rank"],
        "preserved_original_wavefield_files": len(original_hashes),
        "preserved_baseline_zip_sha256": hashlib.sha256((OUTPUT / "baseline3072.zip").read_bytes()).hexdigest(),
        "cpu_lock_sha256": baseline_manifest["requirements.lock"],
        "caveats": [
            "One before/after run on a shared laptop; machine load, JIT shape and cache effects differ, not a controlled isolated speedup campaign.",
            "Sum of rank peak RSS is not simultaneous node peak; concurrent descendant RSS sampling is not unique physical RAM.",
            "No device-transfer speedup inferred from CPU NumPy/JAX buffer aliasing.",
            "Original surface previews remain unchanged because fields match at FP64 tolerance.",
            "Rainier remains geometry-only with unassigned geology/materials/source; no150k propagation or JIT benchmark.",
        ],
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
