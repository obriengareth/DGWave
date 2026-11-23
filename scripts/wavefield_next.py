"""One further fixed-geometry mesh level, with bounded four-core resource checks."""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from dg_wave.runtime import configure_cpu_environment

configure_cpu_environment()

import numpy as np

from dg_wave.attenuation import fit_relaxation
from dg_wave.config import load_config
from dg_wave.materials import assign_materials, sponge_rates
from dg_wave.mesh import read_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.run import receiver_interpolation
from dg_wave.solver import conservative_timestep
from dg_wave.surface import field_comparison, free_surface_mask, refine_tetrahedra, surface_quadrature
from scripts.wavefield_study import OUTPUT, ROOT, TIMES, write_mesh


BUDGET_MIB = 4096
HOST_RESERVE_MIB = 1536
WSL_RESERVE_MIB = 2048
POWERSHELL = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def memory_available():
    host = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command",
         "[double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    windows = float(host.stdout.strip().lstrip("\ufeff"))
    linux = next(float(line.split()[1])/1024 for line in Path("/proc/meminfo").read_text().splitlines()
                 if line.startswith("MemAvailable:"))
    return {"windows_available_mib": windows, "wsl_available_mib": linux}


def process_rss(pid):
    pending, seen, rss = [pid], set(), 0.0
    while pending:
        child = pending.pop()
        if child in seen:
            continue
        seen.add(child)
        try:
            directory = Path(f"/proc/{child}")
            status = (directory / "status").read_text()
            rss += next((float(s.split()[1])/1024 for s in status.splitlines()
                         if s.startswith("VmRSS:")), 0)
            pending.extend(map(int, (directory / "task" / str(child) / "children").read_text().split()))
        except FileNotFoundError:
            continue
    return rss


def prepare():
    memory = memory_available()
    if min(memory.values()) < BUDGET_MIB + HOST_RESERVE_MIB:
        raise RuntimeError(f"insufficient memory headroom for {BUDGET_MIB}MiB job budget: {memory}")
    old = json.loads((OUTPUT / "comparison.json").read_text())
    if "spatial_fine_veryfine" in old:
        raise ValueError("next-level comparison already exists; refusing to overwrite the completed study")
    fine = read_mesh(OUTPUT / "fine.msh")
    mesh, parents = refine_tetrahedra(fine)
    if len(fine.cells) != 3072 or len(mesh.cells) != 24576:
        raise ValueError("expected 3072 parents and exactly 24576 conforming children")
    a, b = prepare_mesh(fine, 2), prepare_mesh(mesh, 2)
    np.testing.assert_array_equal(mesh.physical_tags, fine.physical_tags[parents])
    np.testing.assert_allclose(b.geometry.volumes*8, a.geometry.volumes[parents], rtol=1e-12)
    selected = free_surface_mask(b, (10,))
    _, areas = surface_quadrature(b, selected)
    with np.load(OUTPUT / "sampling.npz") as samples:
        points, weights = samples["points"], samples["weights"]
        rows, interpolation = receiver_interpolation(b, points)
        reconstructed = np.einsum("ri,rij->rj", interpolation, b.geometry.nodes[rows])
        error = float(np.max(np.abs(reconstructed-points)))
        np.testing.assert_allclose(areas.sum(), weights.sum(), rtol=1e-12)
    if error > 1e-9:
        raise ValueError("common original surface samples do not lie on new mesh geometry")
    write_mesh(OUTPUT / "veryfine.msh", mesh)
    np.savez(OUTPUT / "next-refinement.npz", parent_rows=parents, parent_ids=fine.element_ids,
             child_ids=mesh.element_ids)
    original = load_config(OUTPUT / "fine.toml")
    material = assign_materials(mesh, original)
    relaxation = fit_relaxation(material, original.attenuation, physical_tags=mesh.physical_tags)
    bounds = (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
    damping = sponge_rates(b, original.sponge, bounds)
    limit = conservative_timestep(b, relaxation.unrelaxed, damping=damping+relaxation.rate_limit)
    elastic_limit = conservative_timestep(b, relaxation.unrelaxed, damping=damping)
    dt, steps = original.dt, original.steps
    while dt > limit:
        dt /= 2
        steps *= 2
    text = (OUTPUT / "fine.toml").read_text()
    changed = text.replace(f"dt = {original.dt}", f"dt = {dt}").replace(
        f"steps = {original.steps}", f"steps = {steps}")
    veryfine = changed.replace(str(original.mesh), str(OUTPUT / "veryfine.msh"))
    (OUTPUT / "veryfine.toml").write_text(veryfine)
    (OUTPUT / "fine_time_control.toml").write_text(changed)
    (OUTPUT / "veryfine_pilot.toml").write_text(veryfine.replace(f"steps = {steps}", "steps = 64"))
    for level in ("veryfine", "fine_time_control"):
        config = load_config(OUTPUT / f"{level}.toml")
        for key in ("source", "attenuation", "material", "materials", "boundaries", "free_surface_tags"):
            if asdict(config).get(key) != asdict(original).get(key):
                raise ValueError(f"{level} changed physical configuration: {key}")
    fine_metrics = old["runs"]["fine"]
    estimate = 8*(steps/original.steps)*fine_metrics["solve_seconds_max_rank"]
    report = {
        "initial_memory": memory, "job_rss_budget_mib": BUDGET_MIB,
        "minimum_host_available_mib": HOST_RESERVE_MIB,
        "minimum_wsl_available_mib": WSL_RESERVE_MIB,
        "cells": len(mesh.cells), "expected_owned_cells_per_rank": 6144,
        "degree": 2, "dg_nodes": len(mesh.cells)*10, "physical_unknowns": len(mesh.cells)*90,
        "full_state_unknowns": len(mesh.cells)*270, "free_faces": int(selected.sum()),
        "dt_s": dt, "steps": steps, "duration_s": dt*steps,
        "host_global_dt_limit_s": limit, "unrelaxed_elastic_sponge_limit_s": elastic_limit,
        "relaxation_rate_limit_s_inverse": float(relaxation.rate_limit),
        "initial_linear_solve_estimate_s": estimate,
        "cost_caveat": "8x cells and possibly more steps; cache/memory effects require the measured64-step MPI4 pilot",
        "geometry_coordinate_error_m": error, "surface_area_m2": float(areas.sum()),
        "geometry": "second conforming1-to-8 subdivision of original affine cells; terrain and material interfaces fixed",
        "comparison_sampling": "unchanged1536 original physical quadrature points and8 saved times; no resampling",
    }
    (OUTPUT / "next-preflight.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"preflight": report}), flush=True)
    return report


def execute(level, times, timeout, expected_rss_mib=BUDGET_MIB, *, expected_cells=None):
    path = OUTPUT / level
    if path.exists():
        raise ValueError(f"refusing to overwrite an existing run directory: {path}")
    memory = memory_available()
    if (memory["windows_available_mib"] < expected_rss_mib+HOST_RESERVE_MIB
            or memory["wsl_available_mib"] < expected_rss_mib+WSL_RESERVE_MIB):
        raise RuntimeError(f"memory headroom has fallen before {level}: {memory}")
    command = [
        "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core", "-n", "4",
        sys.executable, "-m", "dg_wave.wavefield", str(OUTPUT / f"{level}.toml"),
        str(OUTPUT / "sampling.npz"), str(path), "--times", *map(str, times),
    ]
    started = time.perf_counter()
    peak, host_min, wsl_min = 0., memory["windows_available_mib"], memory["wsl_available_mib"]
    next_host = started+30
    with (OUTPUT / f"{level}.stdout").open("w") as out, (OUTPUT / f"{level}.stderr").open("w") as err:
        process = subprocess.Popen(command, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                peak = max(peak, process_rss(process.pid))
                now = time.perf_counter()
                if now >= next_host:
                    memory = memory_available()
                    host_min = min(host_min, memory["windows_available_mib"])
                    wsl_min = min(wsl_min, memory["wsl_available_mib"])
                    next_host = now+30
                if host_min < HOST_RESERVE_MIB or wsl_min < WSL_RESERVE_MIB or peak > BUDGET_MIB:
                    raise RuntimeError(f"resource guard stopped {level}: host={host_min}MiB WSL={wsl_min}MiB aggregate RSS={peak}MiB")
                if now-started > timeout:
                    raise TimeoutError(f"{level} exceeded {timeout:.0f}s protected wall budget")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    wall = time.perf_counter()-started
    if process.returncode:
        raise RuntimeError(f"{level} exited{process.returncode}; see {OUTPUT / (level+'.stderr')}")
    metrics_path = path / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    cells = expected_cells if expected_cells is not None else (3072 if level == "fine_time_control" else 24576)
    if metrics["elements"] != cells or [r["elements"] for r in metrics["ranks"]] != [cells//4]*4:
        raise ValueError("actual partition/cell counts differ from the planned balanced mesh")
    if len({tuple(r["socket_core"]) for r in metrics["ranks"]}) != 4:
        raise ValueError("rank physical-core bindings are not distinct")
    metrics.update(
        launcher_wall_seconds=wall, launcher_command=command,
        launcher_wall_method="external perf_counter around monitored mpiexec through exit; includes startup/setup/JIT/output, <=1s RSS sampling and30s host-memory queries",
        sampled_aggregate_process_rss_peak_mib=peak,
        sampled_aggregate_process_rss_method="1s sum of VmRSS across this launcher's process descendants, including launcher; sampled concurrent process RSS, not unique physical RAM/whole-node peak",
        windows_available_min_sampled_mib=host_min, wsl_available_min_sampled_mib=wsl_min,
        start_expected_job_rss_mib=expected_rss_mib,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"completed": level, "metrics": metrics}), flush=True)
    return metrics


def analyse(report, metrics):
    path = OUTPUT / "comparison.json"
    original = path.read_bytes()
    if (OUTPUT / "comparison-two-level.json").exists():
        raise ValueError("archived original comparison already exists; refusing replacement")
    old = json.loads(original)
    arrays = {}
    for level in ("fine", "veryfine", "fine_time_control"):
        if level == "fine_time_control" and level not in metrics:
            continue
        with np.load(OUTPUT / level / "surface.npz") as data:
            arrays[level] = {key: data[key].copy() for key in
                             ("sampled_velocity", "comparison_points", "area_weights", "times_s")}
    reference = arrays["veryfine"]
    for actual in arrays.values():
        for key in ("comparison_points", "area_weights", "times_s"):
            np.testing.assert_array_equal(actual[key], reference[key])
    weights, times = reference["area_weights"], reference["times_s"]
    spatial = field_comparison(arrays["fine"]["sampled_velocity"], reference["sampled_velocity"], weights, times)
    control = None
    if "fine_time_control" in arrays:
        temporal = field_comparison(arrays["fine"]["sampled_velocity"],
                                    arrays["fine_time_control"]["sampled_velocity"], weights, times)
        ratio = float(np.sqrt(temporal["spacetime_difference_norm_squared"]/spatial["spacetime_difference_norm_squared"]))
        if ratio > .01 or temporal["spacetime_relative_l2"] > .001:
            raise RuntimeError(f"new time-discretisation control failed: {temporal}, ratio={ratio}")
        common_dt = field_comparison(arrays["fine_time_control"]["sampled_velocity"],
                                    reference["sampled_velocity"], weights, times)
        control = {"fine_relative_temporal": temporal, "temporal_over_new_spatial_norm": ratio,
                   "fine_veryfine_common_dt": common_dt}
    old.update(
        description="three fixed-geometry mesh levels; measured waveform differences, not an established convergence order or exact error",
        levels=["coarse", "fine", "veryfine"], spatial_fine_veryfine=spatial,
        next_time_control=control, next_preflight=report,
    )
    old["runs"].update(metrics)
    old["limitations"][0] = "Three levels on fixed samples show mesh sensitivity, not an established asymptotic order or proof of resolution."
    old["limitations"].append("Original1536 quadrature points retained for identical comparison measure; integration has not been independently refined for the finest field.")
    (OUTPUT / "comparison-two-level.json").write_bytes(original)
    path.write_text(json.dumps(old, indent=2))
    np.savez_compressed(OUTPUT / "comparison-next.npz", points=reference["comparison_points"],
                        area_weights=weights, times_s=times,
                        **{level+"_velocity": data["sampled_velocity"] for level, data in arrays.items()})
    from dg_wave.wavefield_view import write_view
    write_view(OUTPUT, old)
    print(json.dumps({"new_spatial": spatial, "time_control": control}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--resume-after-pilot", action="store_true")
    args = parser.parse_args()
    report = json.loads((OUTPUT / "next-preflight.json").read_text()) if args.resume_after_pilot else prepare()
    if args.prepare_only:
        return
    if args.resume_after_pilot:
        pilot = json.loads((OUTPUT / "veryfine_pilot" / "metrics.json").read_text())
        config = load_config(OUTPUT / "veryfine.toml")
        if (pilot["elements"] != 24576 or pilot["steps"] != 64 or pilot["dt_s"] != config.dt
                or config.dt != report["dt_s"] or config.steps != report["steps"]
                or not np.isclose(pilot["global_dt_limit_s"], report["host_global_dt_limit_s"], rtol=1e-12)):
            raise ValueError("completed pilot/config do not match the prepared next-level experiment")
    else:
        pilot = execute("veryfine_pilot", [0., 64*report["dt_s"]], 300)
    estimate = pilot["solve_seconds_max_rank"]*report["steps"]/64
    report["pilot_linear_solve_estimate_s"] = estimate
    expected_rss = pilot["sum_rank_peak_rss_mib"]+256
    report["post_pilot_expected_full_job_rss_mib"] = expected_rss
    report["post_pilot_budget_method"] = "measured pilot sum of rank peaks plus256MiB margin for additional snapshots; unchanged live4096MiB ceiling and host/WSL reserves"
    (OUTPUT / "next-preflight.json").write_text(json.dumps(report, indent=2))
    if estimate > 6000 or pilot["sum_rank_peak_rss_mib"] > BUDGET_MIB-256:
        raise RuntimeError(f"pilot predicts unsafe/over-budget full run: solve{estimate:.1f}s, rank peak sum{pilot['sum_rank_peak_rss_mib']:.1f}MiB")
    metrics = {}
    if report["dt_s"] != load_config(OUTPUT / "fine.toml").dt:
        old = json.loads((OUTPUT / "fine" / "metrics.json").read_text())
        metrics["fine_time_control"] = execute("fine_time_control", TIMES, 1200,
                                              old["sum_rank_peak_rss_mib"]+256)
    metrics["veryfine"] = execute("veryfine", TIMES, min(7200, 2*estimate+120), expected_rss)
    analyse(report, metrics)


if __name__ == "__main__":
    main()
