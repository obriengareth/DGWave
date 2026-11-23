"""Algebra/config/I-O checks only: no DG residual, RK steps, pilot or wave run."""

from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from dg_wave.checkpoint import _metadata, chunk_lengths
from dg_wave.config import Source, load_config
from dg_wave.geometry import build_geometry
from dg_wave.mesh import box_mesh
from dg_wave.reference import reference_operators
from dg_wave.sources import (
    compact_gaussian, double_couple_tensor, moment_fraction, moment_rate,
    moment_source_values, spatial_normalization,
)


@pytest.mark.parametrize("angles", [(0, 90, 90), (25, 45, -90), (130, 70, 0)])
def test_dc_tensor_is_symmetric_trace_free_and_unit_scalar_moment(angles):
    tensor = double_couple_tensor(*angles)
    np.testing.assert_allclose(tensor, tensor.T, atol=1e-14)
    assert np.trace(tensor) == pytest.approx(0, abs=1e-14)
    np.testing.assert_allclose(np.linalg.eigvalsh(tensor), [-1, 0, 1], atol=1e-14)
    if angles == (0, 90, 90):
        np.testing.assert_array_equal(tensor, [[0, 0, 1], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(double_couple_tensor(angles[0], angles[1], angles[2]+180 if angles[2] <= 0 else angles[2]-180), -tensor, atol=1e-14)


def test_causal_moment_rate_units_integral_and_untruncated_tail():
    time = np.linspace(0, 100, 100001)
    frequency = 1/(2*np.pi*1.05)
    rate = moment_rate(time, frequency)
    assert moment_rate(-1, frequency) == moment_rate(0, frequency) == 0
    assert np.min(rate) >= 0
    assert np.trapezoid(rate, time) == pytest.approx(1, abs=1e-13)
    assert moment_fraction(6, frequency) == pytest.approx(.2177084693288437, rel=1e-13)
    assert moment_rate(6, frequency) > 0
    np.testing.assert_allclose(np.gradient(moment_fraction(time[1000:2000], frequency), time[1000:2000])[1:-1],
                               rate[1001:1999], rtol=2e-5, atol=1e-9)


def test_requested_band_keeps_both_endpoints_and_has_normalised_smooth_finite_rate():
    from dg_wave.sources import band_moment_rate_raw, band_temporal_normalization

    source = Source((0, 0, 0), .5, 1e12, 750., 3., band=(.01, 2.), ramp_time=.5,
                    kind="double_couple", moment_angles=(0, 90, 90),
                    waveform="band_moment_rate", duration=6.)
    normalizer = band_temporal_normalization(source)
    assert normalizer == pytest.approx(band_temporal_normalization(source, 256), rel=1e-13)
    np.testing.assert_array_equal(band_moment_rate_raw(np.array([-1., 0., 6., 7.]), source), 0.)
    time = np.linspace(0, 6, 24577)
    rate = band_moment_rate_raw(time, source)/normalizer
    assert np.trapezoid(rate, time) == pytest.approx(1., abs=1e-12)
    assert rate.min() < 0 and rate.max() > 0
    phase = np.linspace(.6, 5.4, 21)
    direct = np.cos(2*np.pi*(phase[:, None]-3)*np.linspace(.01, 2., 64)).mean(axis=1)
    np.testing.assert_allclose(band_moment_rate_raw(phase, source), direct, rtol=1e-13, atol=1e-14)


def test_band_source_callback_supplies_stress_rates_not_a_vertical_force():
    from types import SimpleNamespace
    from dg_wave.run import _source_metadata, smooth_source
    from dg_wave.sources import band_moment_rate_raw, band_temporal_normalization

    nodes = np.array([[[0., 0, 0], [.2, 0, 0], [5., 5, 5]]])
    source = Source((0, 0, 0), .5, 1e12, .5, 3., band=(.01, 2.), ramp_time=.5,
                    kind="double_couple", moment_angles=(0, 90, 90), normalization=1e9,
                    waveform="band_moment_rate", duration=6.)
    source = replace(source, temporal_normalization=band_temporal_normalization(source))
    callback = smooth_source(
        SimpleNamespace(geometry=SimpleNamespace(nodes=nodes)),
        SimpleNamespace(source=source, boundaries=("absorbing",)*3),
        domain_bounds=(np.full(3, -10), np.full(3, 10)),
    )
    for time in (0., .5, 3., 6., 6.0001, 8., 10.):
        actual = np.asarray(callback(time))
        expected = moment_source_values(nodes, source)*band_moment_rate_raw(time, source)/source.temporal_normalization
        np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-10)
        assert actual.shape[-1] == 9 and np.all(actual[..., :3] == 0)
        if time >= 6.:
            np.testing.assert_array_equal(actual, np.zeros_like(actual))
    assert np.max(np.abs(callback(3.)[..., 7])) > 0
    metadata = _source_metadata(source, 6.)
    assert metadata["cumulative_fraction_at_run_end"] == 1.
    assert "NET scalar moment" in metadata["amplitude_units"]


def test_discrete_moment_normalization_stress_sign_and_ownership_algebra():
    from types import SimpleNamespace

    mesh = box_mesh(2, (4., 4., 4.))
    ref = reference_operators(2)
    geo = build_geometry(mesh, ref)
    prepared = SimpleNamespace(mesh=mesh, reference=ref, geometry=geo)
    source = Source((2, 2, 2), .1, 7e8, .6, 0,
                    kind="double_couple", moment_angles=(0, 90, 90))
    source = replace(source, normalization=spatial_normalization(prepared, source))
    forcing = moment_source_values(geo.nodes, source)
    integrated = np.einsum("eic,i,e->c", forcing, ref.mass.sum(axis=0), geo.determinants)
    expected = np.zeros(9)
    expected[7] = -source.amplitude
    np.testing.assert_allclose(integrated, expected, rtol=1e-14, atol=1e-6)
    partial = sum(np.einsum("eic,i,e->c", forcing[rows], ref.mass.sum(axis=0), geo.determinants[rows])
                  for rows in np.array_split(np.arange(len(mesh.cells)), 4))
    np.testing.assert_allclose(partial, expected, rtol=1e-14, atol=1e-6)
    assert np.all(forcing[..., :3] == 0)
    assert compact_gaussian([[2, 2, 3.8], [2, 2, 4]], source.position, source.width).max() < 1e-15


@pytest.mark.parametrize(("dt", "total", "trace", "surface"), [
    (.00025, 24000, 16, 400), (.004/17, 25500, 17, 425),
])
def test_exact_dual_output_schedule_without_advancing_a_state(dt, total, trace, surface):
    current = 0
    receivers, surfaces = [0], [0]
    for count in chunk_lengths(0, total, trace, surface):
        current += count
        if current % trace == 0:
            receivers.append(current)
        if current % surface == 0:
            surfaces.append(current)
    assert current == total and len(receivers) == 1501 and len(surfaces) == 61
    np.testing.assert_allclose(np.asarray(receivers)*dt, np.arange(1501)*.004, atol=1e-15, rtol=0)
    np.testing.assert_allclose(np.asarray(surfaces)*dt, np.arange(61)*.1, atol=1e-15, rtol=0)
    assert list(chunk_lengths(7, 17, 4, 10)) == [1, 2, 2, 4, 1]


def test_source_and_output_configuration_is_in_checkpoint_fingerprint():
    from dg_wave.config import parse_config, tomllib

    root = Path(__file__).parents[1]
    with (root/"attenuating_box.toml").open("rb") as stream:
        data = tomllib.load(stream)
    data["source"] = dict(position=[500, 500, 350], frequency=.1, amplitude=1e12, width=50.,
                          kind="double_couple", strike=0, dip=90, rake=90, normalization=1e6)
    data["output"].update(surface_interval=400, receiver_components=3)
    config = parse_config(data, root)
    metadata = _metadata(config)
    assert metadata["source"]["moment_angles"] == (0., 90., 90.)
    assert metadata["source"]["normalization"] == 1e6
    assert metadata["surface_interval"] == 400 and metadata["receiver_components"] == 3
    with pytest.raises(ValueError, match="requires width"):
        del data["source"]["normalization"]
        parse_config(data, root)
    legacy = _metadata(load_config(root/"attenuating_box.toml"))
    assert "surface_interval" not in legacy and "receiver_components" not in legacy
    assert not {"kind", "normalization", "moment_angles"} & set(legacy["source"])


def test_velocity_only_csv_and_surface_files_without_physics(tmp_path):
    from dg_wave.run import write_histories
    from dg_wave.operators import prepare_mesh
    from dg_wave.snapshots import SurfaceOutput

    root = Path(__file__).parents[1]
    config = replace(load_config(root/"elastic_box.toml"), output=tmp_path, receivers=((.2, .2, .2),),
                     receiver_components=3, surface_interval=400)
    values = np.arange(9)[None, :]
    write_histories(config, [0.], [values], [0.], [42])
    with (tmp_path/"receivers.csv").open() as stream:
        reader = csv.DictReader(stream)
        row = next(reader)
    assert set(row) == {"time_s", "receiver", "element_id", "x_m", "y_m", "z_m", "vx", "vy", "vz"}
    prepared = prepare_mesh(box_mesh(), 2)
    mask = prepared.connectivity.neighbours < 0
    output = SurfaceOutput(prepared, mask, config)
    state = np.broadcast_to(np.arange(39, dtype=float), (len(prepared.mesh.cells), 10, 39)).copy()
    # Only a tiny selected-face interpolation is evaluated, not a DG physics kernel.
    output.record(state, 0)
    with np.load(tmp_path/"surface"/"step00000000.npz") as arrays:
        velocity = arrays["velocity_m_s"]
        assert velocity.shape == (np.count_nonzero(mask), 6, 3)
        np.testing.assert_allclose(velocity, np.broadcast_to(np.arange(3), velocity.shape), atol=1e-13)
    resumed = SurfaceOutput(prepared, mask, config, restart_step=0)
    resumed.record(state, 400)
    manifest = json.loads((tmp_path/"surface"/"manifest.json").read_text())
    assert manifest["steps"] == [0, 400]
    with pytest.raises(ValueError, match="overwrite"):
        SurfaceOutput(prepared, mask, config)


def test_superseded_source_proposal_cannot_launch_any_process(tmp_path, monkeypatch):
    from scripts import rainier_launch

    (tmp_path/"approval.json").write_text(json.dumps({
        "run_blocked": True, "blocked_reason": "requested band requires mesh decision",
    }))
    monkeypatch.setattr(rainier_launch, "SETUP", tmp_path)
    monkeypatch.setattr(rainier_launch.sys, "argv", ["rainier_launch", "--approve-run"])

    def forbidden(*args, **kwargs):
        raise AssertionError("a superseded draft tried to start a subprocess")

    monkeypatch.setattr(rainier_launch.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError, match="requires mesh decision"):
        rainier_launch.main()


def test_live_progress_writes_only_root_and_labels_actual_model_time(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from dg_wave.mpi_run import _live_progress

    path = tmp_path/"progress.json"
    monkeypatch.setenv("DG_WAVE_LIVE_PROGRESS", str(path))
    config = SimpleNamespace(steps=25500, dt=.004/17)
    _live_progress(SimpleNamespace(rank=0), config, "solve", 425, trace_samples=26, surface_snapshots=2)
    data = json.loads(path.read_text())
    assert data["step"] == 425 and data["simulation_time_s"] == pytest.approx(.1)
    assert data["trace_samples"] == 26 and data["surface_snapshots"] == 2
    _live_progress(SimpleNamespace(rank=1), config, "solve", 850)
    assert json.loads(path.read_text()) == data
