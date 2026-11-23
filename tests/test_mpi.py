import csv
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

import meshio
import numpy as np
import pytest


@pytest.fixture(scope="module")
def mpi_results(tmp_path_factory):
    reports = {}
    worker = Path(__file__).with_name("mpi_worker.py")
    for ranks in (1, 2, 4):
        output = tmp_path_factory.mktemp(f"mpi-{ranks}")
        result = subprocess.run(
            [
                "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
                "-n", str(ranks), sys.executable, str(worker), str(output),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        reports[ranks] = (json.loads(result.stdout), output)
    return reports


@pytest.mark.parametrize("ranks", [1, 2, 4])
def test_actual_mpi_halo_order_and_affinity(mpi_results, ranks, record_property):
    report, _ = mpi_results[ranks]
    record_property("mpi_report", json.dumps(report))
    assert report["mpi_size"] == ranks
    assert len(report["affinities"]) == ranks
    assert report["cases"][2]["peer_counts"] == [0] * ranks
    for case in report["cases"]:
        assert case["halo_exchanges"] == [24] * ranks
        assert case["field_scaled_error"] <= 1e-10
        assert case["receiver_scaled_error"] <= 1e-10
        assert case["surface_scaled_error"] <= 1e-10
    if ranks > 1:
        assert report["cases"][2]["surface_local_face_counts"] == [1]+[0]*(ranks-1)


@pytest.mark.parametrize("kind", ["periodic", "absorbing", "isolated", "layered", "attenuating"])
def test_one_two_four_rank_fields_and_receivers_agree(mpi_results, kind):
    baseline = np.load(mpi_results[1][1] / f"{kind}.npz")
    for ranks in (2, 4):
        if kind == "attenuating" and ranks == 2:
            continue
        actual = np.load(mpi_results[ranks][1] / f"{kind}.npz")
        for key in ("element_ids", "receiver_element_ids"):
            np.testing.assert_array_equal(actual[key], baseline[key])
        for key in ("state", "receivers"):
            np.testing.assert_allclose(actual[key], baseline[key], atol=1e-10, rtol=1e-10)


@pytest.fixture(scope="module")
def cli_results(tmp_path_factory):
    root = Path(__file__).parents[1]
    directory = tmp_path_factory.mktemp("mpi-cli")
    config = (root / "elastic_box.toml").read_text()
    config = config.replace('path = "elastic_box.vtu"', f'path = "{(root / "elastic_box.vtu").as_posix()}"')
    config = config.replace('directory = "output/elastic_box"', 'directory = "result"')
    config = config.replace("steps = 750", "steps = 8").replace("interval = 10", "interval = 3")
    path = directory / "input.toml"
    path.write_text(config)
    reports = {}
    for ranks in (1, 2, 4):
        result = subprocess.run(
            [
                "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
                "-n", str(ranks), sys.executable, "-m", "dg_wave", "run", str(path),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        reports[ranks] = json.loads(result.stdout)
    return directory, path, reports


@pytest.mark.parametrize("ranks", [2, 4])
def test_distributed_cli_fields_receivers_and_visualisation(cli_results, ranks):
    directory, _, reports = cli_results
    serial = directory / "result"
    distributed = serial / f"mpi{ranks}"
    with np.load(serial / "final.npz") as reference, np.load(distributed / "final.npz") as actual:
        np.testing.assert_array_equal(actual["element_ids"], reference["element_ids"])
        np.testing.assert_allclose(actual["state"], reference["state"], atol=1e-10, rtol=1e-10)
    traces = []
    for output in (serial, distributed):
        with (output / "receivers.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        traces.append(np.array([
            [float(row[name]) for name in (
                "time_s", "receiver", "element_id", "vx", "vy", "vz",
                "sxx", "syy", "szz", "sxy", "sxz", "syz",
            )] for row in rows
        ]))
    np.testing.assert_allclose(traces[1], traces[0], atol=1e-10, rtol=1e-10)
    pieces = ET.parse(distributed / "final.pvtu").findall(".//Piece")
    assert len(pieces) == ranks
    ids = []
    for piece in pieces:
        mesh = meshio.read(distributed / piece.attrib["Source"])
        ids.extend(mesh.cell_data["element_id"][0])
        assert "global_vertex_id" in mesh.point_data
    assert len(ids) == len(set(ids)) == 384
    records = reports[ranks]["ranks"]
    assert all(record["halo_exchanges"] == 32 for record in records)
    assert len({record["cpu_affinity"][0] for record in records}) == ranks
    assert "final.pvtu" in (distributed / "view.html").read_text()


def test_empty_mpi_partitions_exit_collectively(cli_results):
    directory, path, _ = cli_results
    mesh = meshio.read(Path(__file__).parents[1] / "elastic_box.vtu")
    single = directory / "single.vtu"
    meshio.write_points_cells(single, mesh.points, [("tetra", mesh.cells_dict["tetra"][:1])])
    config = path.read_text().replace(
        str((Path(__file__).parents[1] / "elastic_box.vtu").as_posix()), single.as_posix(),
    )
    invalid = directory / "empty-partitions.toml"
    invalid.write_text(config)
    result = subprocess.run(
        [
            "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
            "-n", "2", sys.executable, "-m", "dg_wave", "run", str(invalid),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "empty partitions unsupported" in result.stderr


@pytest.mark.parametrize("attenuating", [False, True, "spatial"])
def test_layered_cli_serial_four_rank_materials_and_receivers(tmp_path, record_property, attenuating):
    root = Path(__file__).parents[1]
    text = (root / ("attenuating_box.toml" if attenuating else "layered_box.toml")).read_text()
    text = text.replace('path = "layered_box.msh"', f'path = "{(root / "layered_box.msh").as_posix()}"')
    output = "attenuating_box" if attenuating else "layered_box"
    text = text.replace(f'directory = "output/{output}"', 'directory = "layered"')
    text = text.replace("steps = 1400" if attenuating else "steps = 1000", "steps = 40")
    if attenuating == "spatial":
        text += "\n[attenuation.materials.1]\nqp = 60.0\nqs = 40.0\n[attenuation.materials.2]\nqp = 150.0\nqs = 100.0\n"
        text = text.replace(
            'types = ["absorbing", "absorbing", "absorbing"]',
            'types = ["periodic", "absorbing", "absorbing"]',
        ).replace(
            'faces = ["xmin", "xmax", "ymin", "ymax", "zmin"]',
            'faces = ["ymin", "ymax", "zmin"]',
        )
    config = tmp_path / "layered.toml"
    config.write_text(text)
    reports = {}
    for ranks in (1, 4):
        result = subprocess.run(
            [
                "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
                "-n", str(ranks), sys.executable, "-m", "dg_wave", "run", str(config),
            ],
            capture_output=True, text=True, timeout=120, check=True,
        )
        reports[ranks] = json.loads(result.stdout)
    serial, distributed = tmp_path / "layered", tmp_path / "layered" / "mpi4"
    with np.load(serial / "final.npz") as reference, np.load(distributed / "final.npz") as actual:
        np.testing.assert_array_equal(actual["element_ids"], reference["element_ids"])
        np.testing.assert_allclose(actual["state"], reference["state"], atol=1e-10, rtol=1e-10)
        field_error = float(np.max(np.abs(actual["state"] - reference["state"]) / np.maximum(1, np.abs(reference["state"]))))
        if attenuating:
            assert actual["state"].shape[-1] == 27 and np.linalg.norm(actual["state"][..., 9:]) > 0
    receivers = [np.loadtxt(path / "receivers.csv", delimiter=",", skiprows=1) for path in (serial, distributed)]
    np.testing.assert_allclose(receivers[1], receivers[0], atol=1e-10, rtol=1e-10)
    assert reports[4]["material_tags"] == [1, 2]
    assert reports[4]["free_surface"] and reports[4]["sponge"]
    assert sum(rank["heterogeneous_remote_faces"] for rank in reports[4]["ranks"]) > 0
    assert all(rank["material_exchanges"] == 1 for rank in reports[4]["ranks"])
    for rank in range(4):
        mesh = meshio.read(distributed / f"rank{rank:04d}" / "final.vtu")
        tags = mesh.cell_data["material_tag"][0]
        if attenuating:
            assert np.all(mesh.cell_data["vp"][0] > np.where(tags == 1, 4500, 3000))
        else:
            np.testing.assert_allclose(mesh.cell_data["vp"][0], np.where(tags == 1, 4500, 3000))
    if attenuating:
        assert reports[4]["state_components"] == 27 and reports[4]["trace_components"] == 9
    if attenuating == "spatial":
        expected_targets = [{"tag": 1, "qp": 60., "qs": 40.}, {"tag": 2, "qp": 150., "qs": 100.}]
        assert reports[1]["attenuation"]["material_q_targets"] == expected_targets
        assert reports[4]["attenuation"]["material_q_targets"] == expected_targets
        assert reports[1]["conservative_dt_limit"] == reports[4]["conservative_dt_limit"]
        for ranks in (1, 4):
            report = reports[ranks]["attenuation"]
            assert max(report["max_qp_relative_error"], report["max_qs_relative_error"]) <= .1
    record_property("measurement", json.dumps({
        "case": "attenuating CLI serial/four-rank" if attenuating else "layered CLI serial/four-rank",
        "field_scaled_error": field_error,
        "receiver_scaled_error": float(np.max(np.abs(receivers[1] - receivers[0]) / np.maximum(1, np.abs(receivers[0])))),
        "four_rank_report": reports[4],
    }))


@pytest.mark.parametrize("problem", ["missing_material", "unreadable_mesh", "incompatible_q", "incompatible_material_q"])
def test_invalid_layer_setup_exits_collectively(tmp_path, problem):
    root = Path(__file__).parents[1]
    text = (root / "layered_box.toml").read_text()
    text = text.replace('path = "layered_box.msh"', f'path = "{(root / "layered_box.msh").as_posix()}"')
    if problem == "missing_material":
        text = text.replace("[materials.2]", "[materials.3]")
        message = "missing materials [2]"
    elif problem == "unreadable_mesh":
        mesh = tmp_path / "unreadable.msh"
        mesh.write_text("not a Gmsh mesh\n")
        text = text.replace((root / "layered_box.msh").as_posix(), mesh.as_posix())
        message = "meshio could not read mesh"
    else:
        text += '\n[attenuation]\nband = [5.0, 20.0]\nreference_frequency = 10.0\nqp = 1000.0\nqs = 10.0\nmechanisms = 3\n'
        if problem == "incompatible_material_q":
            text = text.replace("qp = 1000.0", "qp = 60.0").replace("qs = 10.0", "qs = 40.0")
            text += "\n[attenuation.materials.2]\nqp = 1000.0\nqs = 10.0\n"
        message = "negative bulk dissipation"
    config = tmp_path / "invalid-layer.toml"
    config.write_text(text)
    result = subprocess.run(
        [
            "mpiexec", "--nooversubscribe", "--bind-to", "core", "--map-by", "core",
            "-n", "2", sys.executable, "-m", "dg_wave", "run", str(config),
        ],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert message in result.stderr
