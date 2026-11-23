import json
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from dg_wave import checkpoint
from dg_wave.config import load_config
from dg_wave.volcano import write_volcano
from dg_wave.config import Source


@pytest.mark.parametrize("change", ["extend", "implicit", "shorten", "dt", "source_duration", "source_normalization", "cadence"])
def test_run_end_extension_changes_no_checkpoint_physics(tmp_path, change):
    config = load_config(Path(__file__).parents[1] / "elastic_box.toml")
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"checkpoint-only mesh fingerprint")
    source = Source((0., 0., 0.), .5, 1e12, 750., 3., band=(.01, 2.), ramp_time=.5,
                    kind="double_couple", moment_angles=(0., 90., 90.), normalization=1e9,
                    waveform="band_moment_rate", duration=6., temporal_normalization=.265)
    config = replace(config, mesh=mesh, source=source, steps=13, output_interval=4,
                     receivers=((0., 0., 0.),))
    ids = np.array([10, 20])
    state = np.arange(2*4*39, dtype=np.float64).reshape(2, 4, 39)
    times = np.array([0, 4, 8, 12, 13])*config.dt
    history = {"times": times, "receivers": np.zeros((5, 1, 9)), "energies": np.ones(5)}
    saved = tmp_path / "checkpoint"
    checkpoint.save_checkpoint(saved, config, ids, state, 13, history)
    pointer = (saved / "complete.json").read_bytes()
    extended = replace(config, steps=20, output=tmp_path / "continuation")
    if change == "shorten":
        extended = replace(extended, steps=12)
    elif change == "dt":
        extended = replace(extended, dt=config.dt*2)
    elif change == "source_duration":
        extended = replace(extended, source=replace(source, duration=10.))
    elif change == "source_normalization":
        extended = replace(extended, source=replace(source, temporal_normalization=.4))
    elif change == "cadence":
        extended = replace(extended, output_interval=5)
    if change != "extend":
        with pytest.raises(ValueError, match="metadata mismatch"):
            checkpoint.load_checkpoint(saved, extended, ids, state.shape, extend_run_end=change != "implicit")
    else:
        recovered, step, recovered_history = checkpoint.load_checkpoint(
            saved, extended, ids, state.shape, extend_run_end=True,
        )
        np.testing.assert_array_equal(recovered, state)
        assert step == 13 and recovered.shape[-1] == 39
        for key in history:
            np.testing.assert_array_equal(recovered_history[key], history[key])
    assert (saved / "complete.json").read_bytes() == pointer


def launch(config, ranks, *args, check=True):
    return subprocess.run(
        ["mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
         "-n", str(ranks), sys.executable, "-m", "dg_wave", "run", str(config), *map(str, args)],
        capture_output=True, text=True, timeout=90, check=check,
    )


@pytest.fixture(scope="module")
def restart_results(tmp_path_factory):
    directory = tmp_path_factory.mktemp("terrain-restart")
    mesh = directory / "terrain.msh"
    write_volcano(mesh, divisions=(2, 2, 2), interface_layer=1)
    text = (Path(__file__).parents[1] / "volcano.toml").read_text()
    text = text.replace('path = "volcano.msh"', f'path = "{mesh}"')
    text = text.replace("steps = 4800", "steps = 40").replace("interval = 48", "interval = 8")
    result = {}
    for ranks in (1, 4):
        run = directory / str(ranks)
        run.mkdir()
        config = run / "input.toml"
        full = run / "full"
        split = run / "split"
        config.write_text(text.replace('directory = "output/volcano"', f'directory = "{full}"'))
        uninterrupted = json.loads(launch(config, ranks).stdout)
        config.write_text(text.replace('directory = "output/volcano"', f'directory = "{split}"'))
        saved = run / "checkpoint"
        launch(config, ranks, "--checkpoint", saved, "--stop-after", 13)
        restarted = json.loads(launch(config, ranks, "--restart", saved).stdout)
        suffix = Path("mpi4") if ranks == 4 else Path(".")
        result[ranks] = {
            "config": config, "checkpoint": saved,
            "full": full / suffix, "split": split / suffix,
            "reports": (uninterrupted, restarted),
        }
    return result


@pytest.mark.parametrize("ranks", [1, 4])
def test_terrain_restart_full_state_and_output_continuity(restart_results, ranks, record_property):
    result = restart_results[ranks]
    with np.load(result["full"] / "final.npz") as full, np.load(result["split"] / "final.npz") as split:
        np.testing.assert_array_equal(full["element_ids"], split["element_ids"])
        assert full["state"].shape[-1] == 27 and np.max(np.abs(full["state"][..., 9:])) > 0
        error = float(np.max(np.abs(split["state"]-full["state"])/np.maximum(1, np.abs(full["state"]))))
        assert error <= 1e-10
    for filename in ("receivers.csv", "energy.csv"):
        histories = [np.loadtxt(result[name] / filename, delimiter=",", skiprows=1) for name in ("full", "split")]
        np.testing.assert_allclose(histories[1], histories[0], atol=1e-10, rtol=1e-10)
        times = histories[1][:, 0]
        unique = np.unique(times)
        np.testing.assert_array_equal(unique, np.arange(0, 41, 8)*0.00025)
        assert np.count_nonzero(times == 0) == (3 if filename == "receivers.csv" else 1)
    a, b = result["reports"]
    assert b["restarted"] and b["steps"] == 40
    assert a["traction_free_faces"] == b["traction_free_faces"] == 8
    assert b["free_surface_tags"] == [10]
    assert 400 < b["resolution_estimate"]["min_s_wavelength_m"] < 600
    record_property("measurement", json.dumps({"case": f"terrain restart ranks={ranks}", "field_scaled_error": error}))


def test_sloping_terrain_serial_four_rank_equivalence(restart_results, record_property):
    paths = [restart_results[r]["full"] for r in (1, 4)]
    with np.load(paths[0] / "final.npz") as a, np.load(paths[1] / "final.npz") as b:
        np.testing.assert_array_equal(a["element_ids"], b["element_ids"])
        error = float(np.max(np.abs(b["state"]-a["state"])/np.maximum(1, np.abs(a["state"]))))
        assert error <= 1e-10
    traces = [np.loadtxt(p / "receivers.csv", delimiter=",", skiprows=1) for p in paths]
    np.testing.assert_allclose(traces[1], traces[0], atol=1e-10, rtol=1e-10)
    report = restart_results[4]["reports"][0]
    assert report["resolution_estimate"] == restart_results[1]["reports"][0]["resolution_estimate"]
    assert sum(r["heterogeneous_remote_faces"] for r in report["ranks"]) > 0
    record_property("measurement", json.dumps({"case": "sloping terrain serial/MPI4", "field_scaled_error": error}))


@pytest.mark.parametrize("problem", ["incomplete", "missing_rank", "checksum", "material", "source_ramp", "process_count", "free_tag"])
def test_incompatible_restart_exits_collectively(restart_results, tmp_path, problem):
    reference = restart_results[4]
    saved = tmp_path / "checkpoint"
    shutil.copytree(reference["checkpoint"], saved)
    pointer = json.loads((saved / "complete.json").read_text())
    directory = saved / pointer["generation"]
    text = reference["config"].read_text()
    ranks, message = 4, "checkpoint"
    if problem == "incomplete":
        (saved / "complete.json").unlink()
    elif problem == "missing_rank":
        (directory / "rank0002.npz").unlink()
    elif problem == "checksum":
        (directory / "rank0001.npz").write_bytes(b"damaged")
        message = "checksum mismatch"
    elif problem == "material":
        text = text.replace("rho = 2400.0", "rho = 2401.0")
        message = "metadata mismatch"
    elif problem == "source_ramp":
        text = text.replace("[source]", "[source]\nramp_time = 0.3")
        message = "metadata mismatch"
    elif problem == "free_tag":
        text = text.replace("free_tags = [10]", "free_tags = [99]")
        message = "free-surface tags are missing"
    else:
        ranks, message = 2, "process count mismatch"
    config = tmp_path / "input.toml"
    config.write_text(text)
    result = launch(config, ranks, "--restart", saved, check=False)
    assert result.returncode != 0 and message in result.stderr


def test_failed_new_generation_keeps_previous_commit(restart_results, tmp_path, monkeypatch):
    saved = tmp_path / "checkpoint"
    reference = restart_results[1]
    shutil.copytree(reference["checkpoint"], saved)
    config = load_config(reference["config"])
    pointer = (saved / "complete.json").read_bytes()
    generation = json.loads(pointer)["generation"]
    with np.load(saved / generation / "rank0000.npz") as arrays:
        ids, state = arrays["element_ids"].copy(), arrays["state"].copy()
    _, step, history = checkpoint.load_checkpoint(saved, config, ids, state.shape)
    atomic = checkpoint._atomic

    def fail_history(path, write):
        if path.name == "history.npz":
            raise OSError("injected disk write failure")
        return atomic(path, write)

    monkeypatch.setattr(checkpoint, "_atomic", fail_history)
    with pytest.raises(ValueError, match="disk write failure"):
        checkpoint.save_checkpoint(saved, config, ids, state*2, step, history)
    assert (saved / "complete.json").read_bytes() == pointer
    recovered, _, _ = checkpoint.load_checkpoint(saved, config, ids, state.shape)
    np.testing.assert_array_equal(recovered, state)


@pytest.mark.parametrize("ranks", [1, 4])
def test_material_q_restart_preserves_all_memory_components(tmp_path, ranks, record_property):
    root = Path(__file__).parents[1]
    mesh = tmp_path / "terrain.msh"
    write_volcano(mesh, divisions=(2, 2, 2), interface_layer=1)
    text = (root / "volcano.toml").read_text()
    text = text.replace('path = "volcano.msh"', f'path = "{mesh}"')
    text = text.replace("steps = 4800", "steps = 16").replace("interval = 48", "interval = 4")
    text += "\n[attenuation.materials.1]\nqp = 60.0\nqs = 40.0\n[attenuation.materials.2]\nqp = 150.0\nqs = 100.0\n"
    config = tmp_path / "input.toml"
    full, split = tmp_path / "full", tmp_path / "split"
    config.write_text(text.replace('directory = "output/volcano"', f'directory = "{full}"'))
    launch(config, ranks)
    config.write_text(text.replace('directory = "output/volcano"', f'directory = "{split}"'))
    saved = tmp_path / "checkpoint"
    launch(config, ranks, "--checkpoint", saved, "--stop-after", 7)
    report = json.loads(launch(config, ranks, "--restart", saved).stdout)
    suffix = Path("mpi4") if ranks == 4 else Path(".")
    with np.load(full / suffix / "final.npz") as a, np.load(split / suffix / "final.npz") as b:
        assert b["state"].shape[-1] == 27 and np.max(np.abs(b["state"][..., 9:])) > 0
        error = float(np.max(np.abs(b["state"]-a["state"])/np.maximum(1, np.abs(a["state"]))))
        assert error <= 1e-10
    for filename in ("receivers.csv", "energy.csv"):
        histories = [np.loadtxt(path / suffix / filename, delimiter=",", skiprows=1) for path in (full, split)]
        np.testing.assert_allclose(histories[1], histories[0], atol=1e-10, rtol=1e-10)
    assert len(report["attenuation"]["material_q_targets"]) == 2
    record_property("measurement", json.dumps({"case": f"spatial Q restart ranks={ranks}", "field_scaled_error": error}))
