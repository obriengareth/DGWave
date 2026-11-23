"""CPU operator microbenchmark, not a wave simulation or mesh-scaling benchmark."""

import argparse
import json
import os
import statistics
import sys
import time

from dg_wave.runtime import configure_cpu_environment


def run_benchmark(degree: int, copies: int, repeats: int) -> dict:
    if type(degree) is not int or degree < 0:
        raise ValueError("degree must be a nonnegative integer")
    if type(copies) is not int or copies < 1 or type(repeats) is not int or repeats < 1:
        raise ValueError("copies and repeats must be positive integers")
    if sys.platform != "linux":
        raise RuntimeError("Stage 1 RSS measurement requires Linux/WSL")
    configure_cpu_environment()
    import resource

    import numpy as np

    from dg_wave.mesh import box_mesh
    from dg_wave.operators import (
        differentiate_host, differentiation_kernel, prepare_mesh,
    )

    prepared = prepare_mesh(box_mesh(), degree)
    points = prepared.geometry.nodes
    base_values = np.sum(points**degree, axis=2)
    values = np.tile(base_values, (copies, 1))
    inverse = np.tile(prepared.geometry.inverse_jacobians, (copies, 1, 1))
    host = np.tile(differentiate_host(base_values, prepared), (copies, 1, 1))
    kernel = differentiation_kernel(degree)
    import jax

    device = jax.devices("cpu")[0]
    device_values = jax.device_put(values, device)
    device_inverse = jax.device_put(inverse, device)
    jax.block_until_ready((device_values, device_inverse))
    start = time.perf_counter()
    compiled = kernel.lower(device_values, device_inverse).compile()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    result = compiled(device_values, device_inverse).block_until_ready()
    warmup_seconds = time.perf_counter() - start
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = compiled(device_values, device_inverse).block_until_ready()
        timings.append(time.perf_counter() - start)
    actual = np.asarray(result)
    scaled_error = float(np.max(np.abs(actual - host) / np.maximum(1.0, np.abs(host))))
    if not np.isfinite(scaled_error) or scaled_error > 1.0e-10:
        raise RuntimeError(f"FP64 host-reference scaled error {scaled_error} exceeds 1e-10")
    if str(result.dtype) != "float64" or {d.platform for d in result.devices()} != {"cpu"}:
        raise RuntimeError("benchmark must execute with CPU float64 arrays")
    return {
        "degree": degree,
        "elements": len(values),
        "nodes_per_element": values.shape[1],
        "fixture_copies": copies,
        "batch_method": "repeat the six-tetrahedron box fields and affine metrics",
        "backend": device.platform,
        "dtype": str(result.dtype),
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "repeats": repeats,
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        "execution_min_seconds": min(timings),
        "execution_median_seconds": statistics.median(timings),
        "execution_max_seconds": max(timings),
        "timing_method": (
            "perf_counter; compile includes tracing/lowering/compilation; one warmup; "
            "each execution includes dispatch and block_until_ready; input transfers excluded"
        ),
        "max_scaled_error": scaled_error,
        "memory": {
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "measurement": (
                "Linux getrusage(RUSAGE_SELF).ru_maxrss in KiB multiplied by 1024; "
                "whole-process cumulative peak, including Python, geometry, arrays and compiler; "
                "not per-kernel or device-only memory"
            ),
            "array_payload_bytes": (
                values.nbytes + inverse.nbytes + prepared.reference.differentiation.nbytes
                + actual.nbytes
            ),
            "array_payload_method": (
                "nbytes of values, inverse Jacobians, differentiation matrices and output, "
                "each counted once; excludes temporary/allocator overhead and duplicate copies"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--copies", type=int, default=1024, help="copies of the six-element fixture")
    parser.add_argument("--repeats", type=int, default=10, help="synchronised timed executions")
    args = parser.parse_args()
    try:
        if hasattr(os, "sched_getaffinity"):
            os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        print(json.dumps(run_benchmark(args.degree, args.copies, args.repeats), indent=2))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"dg-wave benchmark: error: {error}\n")


if __name__ == "__main__":
    main()
