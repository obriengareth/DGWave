"""Root-owned setup/output and local-state MPI elastic solves."""

from dataclasses import replace
import json
import math
import os
import time
from xml.etree import ElementTree as ET

import meshio
import numpy as np

from dg_wave.distributed import DistributedSolver, ReceiverPlan, collective_error
from dg_wave.attenuation import fit_relaxation
from dg_wave.checkpoint import chunk_lengths, history_arrays, load_checkpoint, run_target, save_checkpoint
from dg_wave.mesh import read_mesh
from dg_wave.materials import COEFFICIENTS, assign_materials, sponge_rates
from dg_wave.operators import _jax_runtime, prepare_mesh
from dg_wave.partition import partition_mesh
from dg_wave.physics import wave_speeds
from dg_wave.run import (
    _source_metadata, _write_view, receiver_interpolation, smooth_source, validate_source,
    write_field, write_histories, resolution_estimate,
)
from dg_wave.solver import elastic_connectivity, free_boundary_faces
from dg_wave.runtime import require_cpu_backend

jax = _jax_runtime()
import jax.numpy as jnp


def root_operation(comm, operation):
    result, error = None, None
    if comm.rank == 0:
        try:
            result = operation()
        except (OSError, ValueError, RuntimeError, meshio.ReadError, meshio.WriteError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error)
    if error is not None:
        raise ValueError(error)
    return result


def _partitions(config, size):
    prepared = prepare_mesh(read_mesh(config.mesh), config.degree)
    if len(prepared.mesh.cells) < size:
        raise ValueError(
            f"empty partitions unsupported: {len(prepared.mesh.cells)} tetrahedra for {size} MPI ranks"
        )
    connection = elastic_connectivity(prepared, config.boundaries)
    material = assign_materials(prepared.mesh, config)
    fit_relaxation(material, config.attenuation, physical_tags=prepared.mesh.physical_tags)
    bounds = (prepared.mesh.vertices.min(axis=0), prepared.mesh.vertices.max(axis=0))
    if config.free_surface:
        free_boundary_faces(prepared, connection, bounds, config.free_surface_tags, required=True)
    sponge_rates(prepared, config.sponge, bounds)
    rows, weights = receiver_interpolation(prepared, config.receivers)
    validate_source(prepared, config)
    return partition_mesh(prepared, connection, size, rows, weights, material=material)


def _write_pvtu(path, size, tagged=False):
    vtk = ET.Element("VTKFile", type="PUnstructuredGrid", version="0.1", byte_order="LittleEndian")
    grid = ET.SubElement(vtk, "PUnstructuredGrid", GhostLevel="0")
    points = ET.SubElement(grid, "PPoints")
    ET.SubElement(points, "PDataArray", type="Float64", NumberOfComponents="3")
    point_data = ET.SubElement(grid, "PPointData")
    ET.SubElement(point_data, "PDataArray", type="Int64", Name="global_vertex_id")
    cells = ET.SubElement(grid, "PCellData")
    for name, dtype, components in (
        ("element_id", "Int64", 1), ("mean_velocity", "Float64", 3), ("mean_stress", "Float64", 6),
        ("density", "Float64", 1), ("vp", "Float64", 1), ("vs", "Float64", 1),
    ):
        ET.SubElement(cells, "PDataArray", type=dtype, Name=name, NumberOfComponents=str(components))
    if tagged:
        ET.SubElement(cells, "PDataArray", type="Int64", Name="material_tag", NumberOfComponents="1")
    for rank in range(size):
        ET.SubElement(grid, "Piece", Source=f"rank{rank:04d}/final.vtu")
    ET.ElementTree(vtk).write(path, encoding="utf-8", xml_declaration=True)


def _live_progress(comm, config, stage, step, **fields):
    import os
    import sys
    from pathlib import Path
    from dg_wave.checkpoint import _atomic

    path = os.environ.get("DG_WAVE_LIVE_PROGRESS")
    if comm.rank != 0 or path is None:
        return
    data = {"stage": stage, "step": step, "total_steps": config.steps,
            "simulation_time_s": step*config.dt, **fields}
    try:
        _atomic(Path(path), lambda stream: stream.write(json.dumps(data).encode()))
    except OSError as error:
        print(f"Rainier live progress failed: {error}", file=sys.stderr, flush=True)
        comm.Abort(2)
        raise


def run_distributed(config, comm, *, checkpoint=None, restart=None, stop_after=None, extend_run_end=False):
    require_cpu_backend(config.backend)
    from mpi4py import MPI

    target = root_operation(comm, lambda: run_target(config, stop_after, checkpoint))
    target = comm.bcast(target)
    # Keep existing serial exports and their preview usable.
    config = replace(config, output=config.output / f"mpi{comm.size}")
    _live_progress(comm, config, "mesh/material preprocessing", 0)
    parts = root_operation(comm, lambda: _partitions(config, comm.size))
    partition = comm.scatter(parts)
    local = partition.prepare()
    _live_progress(comm, config, "solver initialisation", 0)
    solver = DistributedSolver(
        partition, config.material, comm, config.boundaries,
        smooth_source(local, config, domain_bounds=partition.bounds),
        free_surface=config.free_surface, sponge=config.sponge,
        attenuation=config.attenuation,
        free_surface_tags=config.free_surface_tags,
    )
    try:
        receivers = ReceiverPlan(partition, comm)
        state = jnp.zeros(solver.kernel.state_shape, dtype=jnp.float64)
        completed = 0
        times, receiver_values, energies = [], [], []
        if restart is not None:
            state, completed, history = load_checkpoint(
                restart, config, local.mesh.element_ids, solver.kernel.state_shape, comm,
                extend_run_end=extend_run_end,
            )
            state = jnp.asarray(state, dtype=jnp.float64)
            if comm.rank == 0:
                times, receiver_values, energies = (list(history[key]) for key in ("times", "receivers", "energies"))
        collective_error(comm, "requested stop step precedes checkpoint step" if target < completed else None)
        snapshots = None
        intervals = () if config.surface_interval is None else (config.surface_interval,)
        if intervals:
            from dg_wave.snapshots import SurfaceOutput

            snapshots = SurfaceOutput(
                local, solver.kernel.free_faces, config, comm=comm,
                restart_step=completed if restart is not None else None,
            )
        solver.validate_advance(state, config.dt, config.steps, 0.0)
        _live_progress(comm, config, "JIT compilation", completed)
        compile_seconds = solver.compile()
        if snapshots is not None:
            start = time.perf_counter()
            snapshots.compile(state)
            compile_seconds += comm.allreduce(time.perf_counter()-start, op=MPI.MAX)

        def record(t, current):
            error = None
            if not np.isfinite(np.asarray(current)).all():
                error = f"nonfinite solution on rank {comm.rank} at time {t:g} s"
            collective_error(comm, error)
            energy = solver.energy(current)
            if not math.isfinite(energy):
                raise RuntimeError(f"nonfinite global energy at time {t:g} s")
            values = receivers.sample(current)
            if comm.rank == 0:
                times.append(t)
                receiver_values.append(values)
                energies.append(energy)

        if restart is None:
            record(0.0, state)
            if snapshots is not None:
                snapshots.record(state, 0)
        solve_seconds = 0.0
        last_progress = time.monotonic()
        _live_progress(comm, config, "solve", completed, jit_seconds=compile_seconds)
        for count in chunk_lengths(completed, target, config.output_interval, *intervals):
            start = time.perf_counter()
            state = solver.advance(state, config.dt, count, completed * config.dt)
            solve_seconds += time.perf_counter() - start
            completed += count
            if completed % config.output_interval == 0 or completed == config.steps:
                record(completed * config.dt, state)
            if snapshots is not None and completed % config.surface_interval == 0:
                snapshots.record(state, completed)
            if time.monotonic()-last_progress >= 5 or completed == target:
                _live_progress(
                    comm, config, "solve", completed, jit_seconds=compile_seconds,
                    trace_samples=len(times),
                    surface_snapshots=len(snapshots.steps) if comm.rank == 0 and snapshots is not None else 0,
                )
                last_progress = time.monotonic()
        _live_progress(comm, config, "checkpoint/final output", completed, jit_seconds=compile_seconds,
                       trace_samples=len(times),
                       surface_snapshots=len(snapshots.steps) if comm.rank == 0 and snapshots is not None else 0)
        collective_error(comm, None if np.isfinite(np.asarray(state)).all() else "nonfinite final/checkpoint state")
        final_energy = solver.energy(state)
        if checkpoint is not None:
            save_checkpoint(
                checkpoint, config, local.mesh.element_ids, state, completed,
                history_arrays(times, receiver_values, energies, len(config.receivers)) if comm.rank == 0 else None,
                comm,
            )
        solve_seconds = comm.allreduce(solve_seconds, op=MPI.MAX)
        rank_output = config.output / f"rank{comm.rank:04d}"
        error = None
        try:
            rank_output.mkdir(parents=True, exist_ok=True)
            write_field(
                rank_output / "final.vtu", local, state, partition.global_vertex_ids,
                material=solver.material,
            )
            np.savez(
                rank_output / "final.npz", element_ids=local.mesh.element_ids,
                state=np.asarray(state), global_vertex_ids=partition.global_vertex_ids,
            )
        except (OSError, ValueError, RuntimeError, meshio.WriteError) as exc:
            error = f"rank {comm.rank} output: {exc}"
        collective_error(comm, error)
        gathered = solver.gather_state(state)
        device = next(iter(state.devices()))
        collective_error(
            comm, None if device.platform == "cpu" and str(state.dtype) == "float64"
            else f"rank {comm.rank} requires CPU float64 arrays",
        )
        local_cp, local_cs = wave_speeds(solver.material)
        heterogeneous_faces = np.any(np.stack([
            getattr(solver.material, name)[:, None] != getattr(solver.kernel.neighbour_material, name)
            for name in COEFFICIENTS
        ], axis=-1), axis=-1) & (partition.remote_ranks >= 0)
        rank_records = comm.gather({
            "rank": comm.rank, "elements": len(local.mesh.cells),
            "peer_count": len(solver.halo.peers),
            "remote_faces": int(np.count_nonzero(partition.remote_ranks >= 0)),
            "halo_exchanges": solver.halo.exchanges, "local_dt_limit": solver.local_timestep,
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "receiver_indices": partition.receiver_indices.tolist(),
            "material_exchanges": solver.halo.material_exchanges,
            "heterogeneous_remote_faces": int(np.count_nonzero(heterogeneous_faces)),
            "material_tags": np.unique(local.mesh.physical_tags).tolist() if local.mesh.physical_tags is not None else [],
            "traction_free_faces": int(np.count_nonzero(solver.kernel.free_faces)),
            "resolution_estimate": resolution_estimate(local, solver.kernel, config.source.band),
            "cp_range": [float(np.min(local_cp)), float(np.max(local_cp))],
            "cs_range": [float(np.min(local_cs)), float(np.max(local_cs))],
            "attenuation": solver.kernel.relaxation.report() if solver.kernel.relaxation is not None else None,
            "elastic_reference_dt_limit": solver.kernel.elastic_timestep,
        })

        def write_root():
            ids, final = gathered
            write_histories(
                config, times, receiver_values, energies, partition.receiver_element_ids,
            )
            np.savez(config.output / "final.npz", element_ids=ids, state=final)
            tagged = local.mesh.physical_tags is not None
            _write_pvtu(config.output / "final.pvtu", comm.size, tagged)
            cp = max(record["cp_range"][1] for record in rank_records)
            cs = max(record["cs_range"][1] for record in rank_records)
            resolution = None
            if config.source.band is not None:
                resolution = dict(rank_records[0]["resolution_estimate"])
                edge = max(r["resolution_estimate"]["max_tetrahedron_edge_m"] for r in rank_records)
                wavelength = min(r["resolution_estimate"]["min_s_wavelength_m"] for r in rank_records)
                resolution.update(
                    max_tetrahedron_edge_m=edge, min_s_wavelength_m=wavelength,
                    min_wavelength_over_max_edge=wavelength/edge,
                    nominal_degree_scaled_samples_per_wavelength=(config.degree+1)*wavelength/edge,
                )
            summary = {
                "elements": len(ids), "degree": config.degree, "steps": completed,
                "configured_steps": config.steps, "restarted": restart is not None,
                "final_time": completed*config.dt, "cp": cp, "cs": cs, "dt": config.dt,
                "conservative_dt_limit": solver.max_timestep,
                "boundaries": list(config.boundaries), "backend": device.platform,
                "free_surface": config.free_surface, "sponge": config.sponge is not None,
                "free_surface_tags": list(config.free_surface_tags), "source_band_hz": config.source.band,
                "traction_free_faces": sum(record["traction_free_faces"] for record in rank_records),
                "resolution_estimate": resolution,
                "attenuation": (
                    {
                        **rank_records[0]["attenuation"],
                        "max_qp_relative_error": max(record["attenuation"]["max_qp_relative_error"] for record in rank_records),
                        "max_qs_relative_error": max(record["attenuation"]["max_qs_relative_error"] for record in rank_records),
                    } if config.attenuation is not None else None
                ),
                "state_components": solver.kernel.state_shape[-1],
                "trace_components": solver.halo.shape[-1],
                "elastic_reference_dt_limit": min(record["elastic_reference_dt_limit"] for record in rank_records),
                "material_tags": sorted({tag for record in rank_records for tag in record["material_tags"]}),
                "source": _source_metadata(config.source, completed*config.dt),
                "receiver_components": config.receiver_components, "surface_interval_steps": config.surface_interval,
                "cp_range": [min(record["cp_range"][0] for record in rank_records), cp],
                "cs_range": [min(record["cs_range"][0] for record in rank_records), cs],
                "dtype": str(state.dtype), "mpi_size": comm.size, "ranks": rank_records,
                "receiver_element_ids": partition.receiver_element_ids.tolist(),
                "compile_seconds": compile_seconds, "solve_seconds": solve_seconds,
                "timing_method": (
                    "maximum rank times; compile includes lowering/compilation; solve includes "
                    "synchronised RK advances and host-buffer MPI, excluding output/diagnostics"
                ),
                "final_energy_J": final_energy, "peak_energy_J": max(energies),
                "max_receiver_speed": (
                    float(np.max(np.linalg.norm(np.asarray(receiver_values)[..., :3], axis=-1)))
                    if len(config.receivers) else 0.0
                ),
                "field_file": "final.pvtu", "output": str(config.output),
            }
            _write_view(config.output / "view.html", config, times, receiver_values, energies, summary)
            (config.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary

        return root_operation(comm, write_root)
    finally:
        solver.close()
