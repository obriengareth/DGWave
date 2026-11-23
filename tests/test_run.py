import csv
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import meshio
import numpy as np
import pytest

from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.run import receiver_interpolation


@pytest.mark.parametrize("distributed", [False, True])
def test_programmatic_gpu_request_cannot_fall_back_to_cpu(distributed):
    from dg_wave.config import load_config
    from dg_wave.mpi_run import run_distributed
    from dg_wave.run import run_simulation

    config = replace(load_config(Path(__file__).parents[1] / "elastic_box.toml"), backend="gpu")
    with pytest.raises(ValueError, match="refusing CPU fallback"):
        if distributed:
            run_distributed(config, None)
        else:
            run_simulation(config)


def test_polynomial_receiver_interpolation():
    prepared = prepare_mesh(box_mesh(), 2)
    positions = np.array([[0.17, 0.28, 0.36], [0.5, 0.5, 0.5]])
    rows, weights = receiver_interpolation(prepared, positions)
    points = prepared.geometry.nodes
    values = points[..., 0]**2 + points[..., 1] * points[..., 2]
    np.testing.assert_allclose(
        np.einsum("ri,ri->r", weights, values[rows]),
        positions[:, 0]**2 + positions[:, 1] * positions[:, 2], atol=1e-10, rtol=1e-10,
    )
    assert prepared.mesh.element_ids[rows[1]] == 100
    with pytest.raises(ValueError, match="outside"):
        receiver_interpolation(prepared, [[2, 0, 0]])


def test_run_outputs(tmp_path):
    root = Path(__file__).parents[1]
    config = (root / "elastic_box.toml").read_text()
    config = config.replace('path = "elastic_box.vtu"', f'path = "{(root / "elastic_box.vtu").as_posix()}"')
    config = config.replace('directory = "output/elastic_box"', 'directory = "result"')
    config = config.replace("steps = 750", "steps = 6").replace("interval = 10", "interval = 4")
    path = tmp_path / "input.toml"
    path.write_text(config)
    result = subprocess.run(
        [sys.executable, "-m", "dg_wave", "run", str(path)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["steps"] == 6
    assert summary["backend"] == "cpu" and summary["dtype"] == "float64"
    assert 0 < summary["dt"] <= summary["conservative_dt_limit"]
    assert summary["compile_seconds"] > 0 and summary["solve_seconds"] > 0
    output = tmp_path / "result"
    with (output / "receivers.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 * 3  # three receivers at t=0, step 4, step 6
    assert np.isfinite([float(row["vx"]) for row in rows]).all()
    snapshot = meshio.read(output / "final.vtu")
    assert snapshot.cell_data["mean_velocity"][0].shape == (384, 3)
    assert "<svg" in (output / "view.html").read_text()


def test_multi_rank_missing_config_reported_collectively():
    result = subprocess.run(
        [
            "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
            "-n", "2", sys.executable, "-m", "dg_wave", "run", "not-loaded.toml",
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "not-loaded.toml" in result.stderr
    assert "No such file" in result.stderr
