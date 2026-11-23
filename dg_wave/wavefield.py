"""Four-core surface-wavefield sampling with separately measured compute/output."""

import argparse
from pathlib import Path
import resource
import sys
import time


def run(args, comm, local_rank, started):
    import json
    import numpy as np
    import jax.numpy as jnp
    from mpi4py import MPI

    from dg_wave.config import load_config
    from dg_wave.distributed import DistributedSolver, ReceiverPlan, SurfacePlan
    from dg_wave.materials import assign_materials
    from dg_wave.mesh import read_mesh
    from dg_wave.mpi_run import root_operation
    from dg_wave.operators import prepare_mesh
    from dg_wave.partition import partition_mesh
    from dg_wave.run import receiver_interpolation, smooth_source, validate_source
    from dg_wave.runtime import physical_binding
    from dg_wave.solver import elastic_connectivity
    from dg_wave.surface import display_triangles, snapshot_steps

    config = root_operation(comm, lambda: load_config(args.input))
    config = comm.bcast(config)
    if comm.size != 4 or config.degree != 2:
        raise ValueError("this initial wavefield experiment requires four MPI ranks and degree 2")
    steps = snapshot_steps(args.times, config.dt, config.steps)
    if steps[0] != 0 or steps[-1] != config.steps:
        raise ValueError("wavefield snapshots must include initial and final steps")

    def setup():
        with np.load(args.sampling, allow_pickle=False) as data:
            points, area = data["points"].copy(), data["weights"].copy()
        prepared = prepare_mesh(read_mesh(config.mesh), config.degree)
        validate_source(prepared, config)
        rows, interpolation = receiver_interpolation(prepared, points)
        connection = elastic_connectivity(prepared, config.boundaries)
        parts = partition_mesh(
            prepared, connection, comm.size, rows, interpolation,
            material=assign_materials(prepared.mesh, config),
        )
        return prepared, parts, points, area

    setup_data = root_operation(comm, setup)
    prepared = setup_data[0] if comm.rank == 0 else None
    partition = comm.scatter(setup_data[1] if comm.rank == 0 else None)
    local = partition.prepare()
    solver = DistributedSolver(
        partition, config.material, comm, config.boundaries,
        smooth_source(local, config, domain_bounds=partition.bounds),
        free_surface=config.free_surface, free_surface_tags=config.free_surface_tags,
        sponge=config.sponge, attenuation=config.attenuation,
    )
    try:
        samplers = ReceiverPlan(partition, comm)
        surface_plan = SurfacePlan(local, solver.kernel.free_faces, comm)
        state = jnp.zeros(solver.kernel.state_shape, dtype=jnp.float64)
        solver.validate_advance(state, config.dt, config.steps, 0)
        setup_seconds = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)
        compilation = solver.compile()
        compilation += surface_plan.compile(state)
        binding = physical_binding(comm, local_rank)
        if str(state.dtype) != "float64" or next(iter(state.devices())).platform != "cpu":
            raise RuntimeError("actual CPU FP64 arrays required")
        samples, faces = [], []
        local_compute, local_output = 0.0, 0.0

        def record():
            nonlocal local_output
            begin = time.perf_counter()
            sampled = samplers.sample(state)
            surface = surface_plan.sample(state)

            def keep():
                if not np.isfinite(surface).all() or not np.isfinite(sampled).all():
                    raise RuntimeError("nonfinite wavefield snapshot")
                samples.append(sampled[:, :3].copy())
                faces.append(surface)
                return None

            root_operation(comm, keep)
            comm.Barrier()
            local_output += time.perf_counter()-begin

        record()
        previous = 0
        for target in steps[1:]:
            begin = time.perf_counter()
            state = solver.advance(state, config.dt, int(target-previous), previous*config.dt)
            local_compute += time.perf_counter()-begin
            previous = int(target)
            record()
        begin = time.perf_counter()
        gathered = solver.gather_state(state)

        def write():
            args.output.mkdir(parents=True, exist_ok=True)
            positions = surface_plan.positions
            np.savez_compressed(
                args.output / "surface.npz", times_s=steps*config.dt,
                face_positions=positions, surface_state=np.asarray(faces),
                triangles=display_triangles(prepared.reference, len(positions)),
                comparison_points=setup_data[2], area_weights=setup_data[3],
                sampled_velocity=np.asarray(samples),
                face_element_ids=surface_plan.element_ids,
                face_indices=surface_plan.face_indices,
            )
            ids, full = gathered
            if not np.isfinite(full).all():
                raise RuntimeError("nonfinite final volume state")
            np.savez_compressed(args.output / "final.npz", element_ids=ids, state=full)
            return None

        root_operation(comm, write)
        comm.Barrier()
        local_output += time.perf_counter()-begin
        compute = comm.allreduce(local_compute, op=MPI.MAX)
        output = comm.allreduce(local_output, op=MPI.MAX)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        thread_count = next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("Threads:"))
        record_data = {
            **binding, "elements": len(local.mesh.cells),
            "scalar_state_dofs": int(np.prod(state.shape)),
            "ru_maxrss_kib": rss, "peak_rss_mib": rss/1024,
            "os_thread_count": thread_count,
            "halo_exchanges": solver.halo.exchanges,
            "halo_remote_trace_shape": solver.halo.shape,
            "free_surface_faces": len(surface_plan.local_faces),
            "volume_gathers": solver.volume_gathers,
            "backend": next(iter(state.devices())).platform, "dtype": str(state.dtype),
            "local_dt_limit": solver.local_timestep,
        }
        ranks = comm.gather(record_data)
        elapsed = comm.allreduce(time.perf_counter()-started, op=MPI.MAX)

        def metrics():
            result = {
                "mpi_processes": comm.size, "elements": sum(r["elements"] for r in ranks),
                "degree": config.degree, "components": state.shape[-1],
                "scalar_state_dofs": sum(r["scalar_state_dofs"] for r in ranks),
                "dt_s": config.dt, "steps": config.steps, "times_s": (steps*config.dt).tolist(),
                "global_dt_limit_s": solver.max_timestep,
                "setup_seconds_max_rank": setup_seconds, "compile_seconds_max_rank": compilation,
                "solve_seconds_max_rank": compute, "output_seconds_max_rank": output,
                "in_process_wall_seconds_max_rank": elapsed,
                "timing_method": "perf_counter; synchronized RK/host-MPI solve; output includes selected surface nodal gathers/sampling and one intentional final full-volume gather/NPZ I/O; maximum rank phase sums; launcher adds full job wall time separately",
                "surface_snapshot_strategy": "JIT selected free-face extraction locally, gather only face nodes/all components, stable global element ID then face; static geometry gathered once",
                "surface_snapshots": len(steps), "intentional_final_volume_gathers": solver.volume_gathers,
                "halo_strategy": "only remote physical9 traces on host/MPI; local neighbour permutation/gather inside shared JAX residual",
                "peak_rss_max_rank_mib": max(r["peak_rss_mib"] for r in ranks),
                "sum_rank_peak_rss_mib": sum(r["peak_rss_mib"] for r in ranks),
                "memory_method": "Linux resource.getrusage(RUSAGE_SELF).ru_maxrss in KiB /1024 = MiB; sum of rank peaks is NOT simultaneous node peak; excludes launcher/orchestrator/previews",
                "ranks": ranks, "source": {
                    "position_m": config.source.position, "direction": config.source.direction,
                    "spatial_width_m": config.source.width, "nominal_band_hz": config.source.band,
                    "ricker_weight_frequency_hz": config.source.frequency, "delay_s": config.source.delay,
                    "ramp_time_s": config.source.ramp_time,
                    "envelope": "quintic smoothstep from zero at t=0 to one at ramp_time; C2; not strictly bandlimited",
                },
                "attenuation_root_rank": solver.kernel.relaxation.report() if solver.kernel.relaxation else None,
            }
            (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result))
            return None

        root_operation(comm, metrics)
    finally:
        solver.close()


def main():
    started = time.perf_counter()
    from dg_wave.runtime import initialize_mpi_cpu

    comm, _, local_rank = initialize_mpi_cpu()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sampling", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--times", type=float, nargs="+", required=True)
    args = parser.parse_args()
    try:
        run(args, comm, local_rank, started)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"dg-wave-wavefield rank {comm.rank}: {error}", file=sys.stderr, flush=True)
        comm.Abort(2)


if __name__ == "__main__":
    main()
