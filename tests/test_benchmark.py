import json
import subprocess
import sys

import numpy as np


def test_synchronised_benchmark_and_memory_report():
    result = subprocess.run(
        [
            sys.executable, "-m", "dg_wave.benchmark",
            "--degree", "2", "--copies", "8", "--repeats", "2",
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    report = json.loads(result.stdout)
    assert report["elements"] == 48
    assert report["nodes_per_element"] == 10
    assert report["backend"] == "cpu"
    assert report["dtype"] == "float64"
    assert report["max_scaled_error"] <= 1.0e-10
    assert len(report["cpu_affinity"]) == 1
    assert report["compile_seconds"] > 0
    assert 0 < report["execution_min_seconds"] <= report["execution_median_seconds"]
    assert np.isfinite(report["execution_max_seconds"])
    expected_payload = 8 * (48 * 10 + 48 * 9 + 3 * 10 * 10 + 48 * 10 * 3)
    assert report["memory"]["array_payload_bytes"] == expected_payload
    assert report["memory"]["process_peak_rss_bytes"] > expected_payload
