"""Short single-Python-process CPU study using the existing DG/RK4 kernels."""

import argparse
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time


def thread_snapshot(allowed):
    result = {}
    for path in Path("/proc/self/task").iterdir():
        try:
            tid = int(path.name)
            stat = (path / "stat").read_text()
            fields = stat[stat.rfind(")")+2:].split()
            mask = sorted(os.sched_getaffinity(tid))
            name = (path / "comm").read_text().strip()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if not mask or not set(mask) <= allowed:
            raise RuntimeError(f"thread{tid} escaped assigned physical-core mask: {mask}")
        result[tid] = {
            "ticks": int(fields[11])+int(fields[12]), "cpu": int(fields[36]), "mask": mask,
            "name": name,
        }
    return result


class ThreadActivity:
    def __init__(self, cpus):
        self.allowed = set(cpus)
        self.stop = threading.Event()
        self.error = None
        self.observed = {}
        self.active_cpu_windows = []
        self.monitor_tid = None

    def monitor(self):
        self.monitor_tid = threading.get_native_id()
        previous = {}
        try:
            while not self.stop.is_set():
                current = thread_snapshot(self.allowed)
                active = set()
                for tid, data in current.items():
                    record = self.observed.setdefault(tid, {"masks": set(), "active_cpus": set()})
                    record["masks"].add(tuple(data["mask"]))
                    if tid in previous and data["ticks"] > previous[tid]["ticks"]:
                        if data["cpu"] not in self.allowed:
                            raise RuntimeError("active kernel thread executed outside the assigned cores")
                        record["active_cpus"].add(data["cpu"])
                        if tid != self.monitor_tid:
                            active.add(data["cpu"])
                self.active_cpu_windows.append(sorted(active))
                previous = current
                self.stop.wait(.05)
        except (OSError, ValueError, RuntimeError) as exc:
            self.error = exc
            self.stop.set()

    def __enter__(self):
        self.before = thread_snapshot(self.allowed)
        self.thread = threading.Thread(target=self.monitor, name="dg-cpu-monitor")
        self.thread.start()
        self.usage = resource.getrusage(resource.RUSAGE_SELF)
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.wall = time.perf_counter()-self.started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu = usage.ru_utime+usage.ru_stime-self.usage.ru_utime-self.usage.ru_stime
        self.stop.set()
        self.thread.join()
        after = thread_snapshot(self.allowed)
        ticks = os.sysconf("SC_CLK_TCK")
        self.records = []
        for tid, data in after.items():
            observed = self.observed.get(tid, {"masks": {tuple(data["mask"])}, "active_cpus": set()})
            self.records.append({
                "tid": tid, "name": data["name"],
                "cpu_seconds_delta": (data["ticks"]-self.before.get(tid, {"ticks": 0})["ticks"])/ticks,
                "observed_affinity_masks": [list(mask) for mask in sorted(observed["masks"])],
                "observed_active_cpus": sorted(observed["active_cpus"]),
            })
        if self.error is not None:
            raise RuntimeError(f"thread activity monitor failed: {self.error}") from self.error

    def report(self):
        return {
            "wall_seconds": self.wall, "process_cpu_seconds": self.cpu,
            "equivalent_occupied_cores": self.cpu/self.wall,
            "threads": self.records, "monitor_tid": self.monitor_tid,
            "max_observed_active_cpu_window": max(map(len, self.active_cpu_windows), default=0),
            "method": "RUSAGE_SELF user+system CPU delta / synchronized wall; /proc thread CPU ticks and masks,50ms samples. Distinct last-CPU observations per window are not instantaneous concurrency; CPU/wall>1 proves concurrent CPU work. Monitor overhead is included; its exited thread is excluded from end-thread table.",
        }


def run(args, binding, started):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from dg_wave.config import load_config
    from dg_wave.materials import assign_materials
    from dg_wave.mesh import read_mesh
    from dg_wave.operators import prepare_mesh
    from dg_wave.run import receiver_interpolation, smooth_source, validate_source
    from dg_wave.solver import ElasticSolver
    from dg_wave.surface import free_surface_mask, surface_trace

    config = load_config(args.input)
    if args.steps < 1 or args.block_steps < 1 or args.steps % args.block_steps or args.repeats < 1:
        raise ValueError("positive steps/repeats and an integral number of positive compile blocks required")
    prepared = prepare_mesh(read_mesh(config.mesh), config.degree)
    validate_source(prepared, config)
    solver = ElasticSolver(
        prepared, assign_materials(prepared.mesh, config), config.boundaries,
        smooth_source(prepared, config), free_surface=config.free_surface,
        free_surface_tags=config.free_surface_tags, attenuation=config.attenuation, sponge=config.sponge,
    )
    solver.validate_timestep(config.dt)
    zero = jnp.zeros(solver.state_shape, dtype=jnp.float64)
    if jax.default_backend() != "cpu" or str(zero.dtype) != "float64":
        raise RuntimeError("actual CPU FP64 required")
    if "mpi4py.MPI" in sys.modules:
        raise RuntimeError("single-process benchmark unexpectedly initialized MPI")
    setup = time.perf_counter()-started
    begin = time.perf_counter()
    kernel = solver.advance_kernel.lower(zero, 0., config.dt, steps=args.block_steps).compile()
    compile_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    kernel(zero, 0., config.dt).block_until_ready()
    warmup_seconds = time.perf_counter()-begin
    measurements = []
    for _ in range(args.repeats):
        state = zero
        with ThreadActivity(binding["logical_cpu_mask"]) as activity:
            for offset in range(0, args.steps, args.block_steps):
                state = kernel(state, offset*config.dt, config.dt)
            state.block_until_ready()
        measurements.append(activity.report())
    begin = time.perf_counter()
    host = np.asarray(state)
    if not np.isfinite(host).all() or np.max(np.abs(host[..., 9:])) == 0:
        raise RuntimeError("finite nonzero full attenuation-memory state required")
    with np.load(args.sampling) as samples:
        rows, interpolation = receiver_interpolation(prepared, samples["points"])
        sampled = np.einsum("ri,ric->rc", interpolation, host[rows, :, :3])
        points = samples["points"].copy()
    mask = free_surface_mask(prepared, config.free_surface_tags)
    positions, surface = surface_trace(prepared, host, mask)
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "final.npz", element_ids=prepared.mesh.element_ids,
                        state=host, sampled_velocity=sampled, comparison_points=points,
                        surface_state=surface, face_positions=positions)
    output_seconds = time.perf_counter()-begin
    result = {
        **binding, "process_count": 1, "mpi_initialized": "mpi4py.MPI" in sys.modules,
        "backend": jax.default_backend(), "dtype": str(state.dtype), "jax_version": jax.__version__,
        "elements": len(prepared.mesh.cells), "degree": config.degree,
        "dg_nodes": len(prepared.mesh.cells)*len(prepared.reference.nodes),
        "physical_unknowns": len(prepared.mesh.cells)*len(prepared.reference.nodes)*9,
        "full_state_unknowns": int(state.size), "dt_s": config.dt,
        "dt_limit_s": solver.max_timestep, "steps": args.steps, "duration_s": args.steps*config.dt,
        "block_steps": args.block_steps, "repeats": args.repeats,
        "setup_seconds": setup, "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds, "solve_seconds_median": float(np.median([m["wall_seconds"] for m in measurements])),
        "output_seconds": output_seconds, "measurements": measurements,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        "memory_method": "Linux RUSAGE_SELF ru_maxrss KiB/1024, single compute Python process",
        "in_process_wall_seconds": time.perf_counter()-started,
        "protocol": f"shared ElasticSolver.advance_kernel,{args.block_steps}-step compiled scan blocks; one untimed block warmup, repeated fresh-zero short solves; no MPI/child workers or separate physics",
        "source": {"position": config.source.position, "ramp_time": config.source.ramp_time,
                   "width": config.source.width, "nominal_band_hz": config.source.band},
    }
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("sampling", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cpus", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--block-steps", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    try:
        from dg_wave.runtime import initialize_threaded_cpu
        binding = initialize_threaded_cpu(args.cpus)
        run(args, binding, started)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"dg-wave-threaded: {exc}\n")


if __name__ == "__main__":
    main()
