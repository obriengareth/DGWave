from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dg_wave.attenuation import fit_relaxation
from dg_wave.config import Attenuation, Material, parse_config, tomllib
from dg_wave.mesh import box_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.solver import ElasticSolver


MATERIAL = Material(1, 8/3, 1)
SETTINGS = Attenuation((5, 20), 10, 60, 40, 3)


def test_passive_fit_and_reference_storage_convention(record_property):
    model = fit_relaxation(MATERIAL, SETTINGS)
    assert np.all(model.bulk >= 0) and np.all(model.shear >= 0)
    bulk, shear = model.moduli(2j*np.pi*SETTINGS.reference_frequency)
    np.testing.assert_allclose(bulk.real[:, 0], MATERIAL.bulk_modulus, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(shear.real[:, 0], MATERIAL.shear_modulus, atol=1e-13, rtol=1e-13)
    bulk, shear = model.moduli(2j*np.pi*np.geomspace(0.01, 1e5, 201))
    assert np.all(bulk.imag >= 0) and np.all(shear.imag > 0)
    assert np.all(model.relaxed.bulk_modulus > 0) and np.all(model.relaxed.shear_modulus > 0)
    report = model.report()
    assert max(report["max_qp_relative_error"], report["max_qs_relative_error"]) < 0.01
    record_property("measurement", json.dumps(report))


@pytest.mark.parametrize(("key", "value"), [
    ("band", [20, 5]), ("band", [0, 20]), ("reference_frequency", 25),
    ("qp", 0), ("qs", float("inf")), ("mechanisms", 0), ("mechanisms", 6),
])
def test_invalid_attenuation_config(key, value):
    with (Path(__file__).parents[1]/"attenuating_box.toml").open("rb") as stream:
        data = tomllib.load(stream)
    data["attenuation"][key] = value
    with pytest.raises(ValueError, match="attenuation"):
        parse_config(data)


def test_incompatible_bulk_loss_is_rejected():
    with pytest.raises(ValueError, match="negative bulk dissipation"):
        fit_relaxation(MATERIAL, replace(SETTINGS, qp=1000, qs=10))


def test_material_q_configuration_and_explicit_global_fallback():
    with (Path(__file__).parents[1]/"attenuating_box.toml").open("rb") as stream:
        data = tomllib.load(stream)
    values = data["attenuation"]
    values["materials"] = {"1": {"qp": 90, "qs": 60}}
    parsed = parse_config(data).attenuation
    assert parsed.material_q == ((1, 90., 60.),)
    del values["qp"], values["qs"]
    with pytest.raises(ValueError, match="every material"):
        parse_config(data)
    values["materials"]["2"] = {"qp": 150, "qs": 100}
    parsed = parse_config(data).attenuation
    assert parsed.qp is None and parsed.qs is None
    assert parsed.material_q == ((1, 90., 60.), (2, 150., 100.))
    values["materials"]["3"] = {"qp": 60, "qs": 40}
    with pytest.raises(ValueError, match="material tags"):
        parse_config(data)


def test_material_q_matches_independent_fits_and_global_limit():
    from dg_wave.materials import COEFFICIENTS, element_material

    tags = np.array([2, 1, 2, 1])
    material = element_material(MATERIAL, len(tags))
    settings = replace(SETTINGS, qp=None, qs=None, material_q=((1, 60., 40.), (2, 150., 100.)))
    model = fit_relaxation(material, settings, physical_tags=tags)
    for tag, qp, qs in settings.material_q:
        independent = fit_relaxation(MATERIAL, replace(SETTINGS, qp=qp, qs=qs))
        for name in ("bulk", "shear", "qp", "qs"):
            actual = getattr(model, name)[tags == tag]
            np.testing.assert_array_equal(actual, np.broadcast_to(getattr(independent, name), actual.shape))
        for kind in ("relaxed", "unrelaxed"):
            for name in COEFFICIENTS:
                np.testing.assert_array_equal(
                    getattr(getattr(model, kind), name)[tags == tag],
                    np.full(np.count_nonzero(tags == tag), getattr(getattr(independent, kind), name)[0]),
                )
    bulk, shear = model.moduli(2j*np.pi*SETTINGS.reference_frequency)
    np.testing.assert_allclose(bulk.real[:, 0], material.bulk_modulus, atol=1e-13)
    np.testing.assert_allclose(shear.real[:, 0], material.shear_modulus, atol=1e-13)
    assert model.report()["material_q_targets"] == [
        {"tag": 1, "qp": 60., "qs": 40.}, {"tag": 2, "qp": 150., "qs": 100.},
    ]
    same = fit_relaxation(
        material, replace(SETTINGS, material_q=((2, 60., 40.),)), physical_tags=tags,
    )
    global_model = fit_relaxation(material, SETTINGS)
    for name in ("bulk", "shear", "qp", "qs"):
        np.testing.assert_array_equal(getattr(same, name), getattr(global_model, name))
    with pytest.raises(ValueError, match="integer physical tag"):
        fit_relaxation(material, settings)
    with pytest.raises(ValueError, match="every element"):
        fit_relaxation(material, replace(settings, material_q=((1, 60., 40.),)), physical_tags=tags)
    with pytest.raises(ValueError, match="negative bulk dissipation"):
        fit_relaxation(material, replace(settings, material_q=((1, 60., 40.), (2, 1000., 10.))), physical_tags=tags)


def test_elastic_limit_and_unused_memory():
    prepared = prepare_mesh(box_mesh(), 1)
    elastic = ElasticSolver(prepared, MATERIAL)
    almost = ElasticSolver(prepared, MATERIAL, attenuation=replace(SETTINGS, qp=6e10, qs=4e10))
    assert elastic.state_shape[-1] == 9 and elastic.relaxation is None
    assert almost.state_shape[-1] == 27
    state = np.random.default_rng(16).normal(size=elastic.state_shape)
    augmented = np.pad(state, ((0, 0), (0, 0), (0, 18)))
    np.testing.assert_allclose(almost.rhs(augmented, 0)[..., :9], elastic.rhs(state, 0), atol=1e-7, rtol=1e-8)


def test_memory_relaxation_passivity_and_timestep():
    prepared = prepare_mesh(box_mesh(), 1)
    solver = ElasticSolver(prepared, MATERIAL, attenuation=SETTINGS)
    assert solver.max_timestep < min(solver.elastic_timestep, np.min(solver.relaxation.tau)/2)
    initial = np.zeros(solver.state_shape)
    branch = np.arange(18).reshape(3, 6)/50
    initial[..., 9:] = branch.ravel()
    initial[..., 3:9] = branch.sum(axis=0)
    final_time = 0.01
    steps = int(np.ceil(final_time/(np.min(solver.relaxation.tau)/20)))
    final = np.asarray(solver.advance(initial, final_time/steps, steps))
    expected = branch*np.exp(-final_time/solver.relaxation.tau[:, None])
    np.testing.assert_allclose(final[..., 9:], np.broadcast_to(expected.ravel(), final[..., 9:].shape), atol=2e-8, rtol=2e-6)
    assert float(solver.energy(final)) < float(solver.energy(initial))
    state = np.random.default_rng(16).normal(size=solver.state_shape)
    assert float(solver.inner_product(state, solver.rhs(state, 0))) < 0
    for step in range(10):
        energy = float(solver.energy(state))
        state = solver.step(state, step*solver.max_timestep, solver.max_timestep)
        assert float(solver.energy(state)) <= energy*(1+1e-12)
    with pytest.raises(ValueError, match="conservative limit"):
        solver.advance(initial, 1.01*solver.max_timestep, 1)


@pytest.mark.parametrize("unique", [False, True])
@pytest.mark.parametrize("spatial_q", [False, True])
def test_reporting_and_resolution_match_full_response_with_bounded_workspace(monkeypatch, unique, spatial_q):
    from dg_wave.attenuation import Relaxation
    from dg_wave.run import resolution_estimate

    count = 270
    variation = np.arange(count)/count if unique else np.zeros(count)
    material = Material(1+.1*variation, 8/3+.2*variation, 1+.05*variation)
    tags = np.arange(count) % 2 + 1
    settings = replace(SETTINGS, material_q=((2, 150., 100.),)) if spatial_q else SETTINGS
    model = fit_relaxation(material, settings, physical_tags=tags)
    frequencies = np.geomspace(*SETTINGS.band, 129)
    bulk, shear = model.moduli(2j*np.pi*frequencies)
    p = bulk+4*shear/3
    expected_errors = [
        np.max(np.abs(p.real/p.imag/model.qp[:, None]-1)),
        np.max(np.abs(shear.real/shear.imag/model.qs[:, None]-1)),
    ]
    expected_wavelength = np.min(1/(frequencies*np.sqrt(model.unrelaxed.density[:, None]/shear).real))
    sizes = []
    original = Relaxation.moduli

    def bounded(self, s, *, elements=slice(None)):
        sizes.append(len(self.bulk[elements]))
        return original(self, s, elements=elements)

    monkeypatch.setattr(Relaxation, "moduli", bounded)
    report = model.report()
    prepared = prepare_mesh(box_mesh(), 1)
    resolution = resolution_estimate(
        prepared, SimpleNamespace(relaxation=model, material=model.unrelaxed), SETTINGS.band,
    )
    np.testing.assert_allclose(
        [report["max_qp_relative_error"], report["max_qs_relative_error"]], expected_errors, atol=1e-14, rtol=1e-13,
    )
    assert resolution["min_s_wavelength_m"] == pytest.approx(expected_wavelength, rel=1e-13)
    assert sizes == [128, 128, 14]*2


@pytest.mark.parametrize("spatial_q", [False, True])
def test_q_checkpoint_metadata_legacy_compatibility_and_changed_target_rejection(tmp_path, spatial_q):
    from dataclasses import asdict
    from dg_wave import checkpoint
    from dg_wave.config import load_config

    config = load_config(Path(__file__).parents[1]/"attenuating_box.toml")
    if spatial_q:
        config = replace(config, attenuation=replace(SETTINGS, material_q=((2, 150., 100.),)))
    ids, state = np.array([42]), np.zeros((1, 4, 27))
    history = checkpoint.history_arrays([0.], [np.zeros((len(config.receivers), 9))], [0.], len(config.receivers))
    checkpoint.save_checkpoint(tmp_path, config, ids, state, 0, history)
    loaded, step, _ = checkpoint.load_checkpoint(tmp_path, config, ids, state.shape)
    np.testing.assert_array_equal(loaded, state)
    assert step == 0
    metadata = checkpoint._metadata(config)
    if not spatial_q:
        legacy = asdict(config.attenuation)
        legacy.pop("material_q")
        assert metadata["attenuation"] == legacy
        altered = replace(config.attenuation, qp=config.attenuation.qp+1)
    else:
        assert metadata["attenuation"]["material_q"] == ((2, 150., 100.),)
        altered = replace(config.attenuation, material_q=((2, 151., 100.),))
    with pytest.raises(ValueError, match="metadata mismatch"):
        checkpoint.load_checkpoint(tmp_path, replace(config, attenuation=altered), ids, state.shape)
