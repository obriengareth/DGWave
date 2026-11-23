"""Bounded1/2/4 physical-core study of one Python process, never an MPI launch."""

import json
from pathlib import Path
import subprocess
import sys
import time

from dg_wave.runtime import configure_cpu_environment, physical_core_cpus


ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "output" / "threaded_cpu"


def main():
    configure_cpu_environment()
    cpus = physical_core_cpus(4)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    for count in (1, 2, 4):
        destination = OUTPUT / f"cores{count}"
        if destination.exists():
            raise ValueError(f"refusing to overwrite {destination}")
        command = [
            sys.executable, "-m", "dg_wave.threaded_benchmark",
            str(ROOT / "output" / "surface_wavefield" / "fine.toml"),
            str(ROOT / "output" / "surface_wavefield" / "sampling.npz"), str(destination),
            "--cpus", *map(str, cpus[:count]),
        ]
        begin = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
        wall = time.perf_counter()-begin
        (OUTPUT / f"cores{count}.stdout").write_text(result.stdout)
        (OUTPUT / f"cores{count}.stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{count}-core child failed: {result.stderr}")
        metrics = json.loads((destination / "metrics.json").read_text())
        metrics.update(launcher_wall_seconds=wall, launcher_command=command,
                       launcher_wall_method="external perf_counter around direct single-Python subprocess through exit")
        (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
        records.append(metrics)
        print(json.dumps({"completed_cores": count, "metrics": metrics}), flush=True)
    import numpy as np

    comparisons = []
    with np.load(OUTPUT / "cores1" / "final.npz") as reference:
        for count in (2, 4):
            with np.load(OUTPUT / f"cores{count}" / "final.npz") as actual:
                np.testing.assert_array_equal(actual["element_ids"], reference["element_ids"])
                np.testing.assert_array_equal(actual["comparison_points"], reference["comparison_points"])
                errors = {}
                for key in ("state", "surface_state", "sampled_velocity"):
                    a, b = actual[key], reference[key]
                    scaled = float(np.max(np.abs(a-b)/np.maximum(1, np.abs(b))))
                    relative = float(np.linalg.norm(a-b)/np.linalg.norm(b))
                    if scaled > 1e-10 or relative > 1e-10:
                        raise RuntimeError(f"{count}-core {key} mismatch: scaled={scaled},relative={relative}")
                    errors[key] = {"scaled_max_error": scaled, "relative_l2": relative}
                a, b = actual["state"][..., 9:], reference["state"][..., 9:]
                memory = float(np.linalg.norm(a-b)/np.linalg.norm(b))
                if memory > 1e-10:
                    raise RuntimeError(f"{count}-core Maxwell-memory relative L2 mismatch: {memory}")
                comparisons.append({"cores": count, "errors": errors, "memory_relative_l2": memory})
    baseline = records[0]["solve_seconds_median"]
    for record in records:
        count = len(record["logical_cpu_mask"])
        record["speedup_vs_1_core"] = baseline/record["solve_seconds_median"]
        record["efficiency_vs_1_core"] = record["speedup_vs_1_core"]/count
        occupied = [m["equivalent_occupied_cores"] for m in record["measurements"]]
        record["observed_occupied_cores_range"] = [min(occupied), max(occupied)]
        record["concurrent_multicore_proven"] = count > 1 and min(occupied) > 1.25
    report = {
        "scope": "Stage7.5: fixed3072-element degree2 viscoelastic problem, one Python process,1/2/4 physical cores",
        "runs": records, "comparisons_vs_1_core": comparisons, "fp64_tolerance": 1e-10,
        "caveats": [
            "Short0.064s window is a per-timestep threading measurement, not a full volcanic propagation run.",
            "Two timed repeats after one64-step warmup; speedup/efficiency refer only to this shared serial JAX scan, not to MPI results.",
            "CPU-time/wall includes small Python sampling/runtime overhead; per-thread deltas/placements show where CPU work occurred.",
            "No MPI launcher or MPI initialization in compute processes; no process-count scaling, new physics, GPU or7.6 work.",
        ],
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"comparison": report}), flush=True)


if __name__ == "__main__":
    main()
