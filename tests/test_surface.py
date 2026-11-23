from dataclasses import replace
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest

from dg_wave.config import load_config
from dg_wave.operators import prepare_mesh
from dg_wave.run import receiver_interpolation, smooth_source
from dg_wave.runtime import check_physical_cores
from dg_wave.surface import (
    display_triangles, field_comparison, free_surface_mask, refine_tetrahedra,
    snapshot_steps, surface_quadrature, surface_trace,
    surface_trace_kernel,
)
from dg_wave.volcano import volcano_mesh


@pytest.mark.parametrize("already_refined", [False, True])
def test_nested_geometry_and_full_dg_surface_polynomial(already_refined):
    coarse = volcano_mesh(divisions=(2, 2, 2), lengths=(1., 1., 1.), height=.3,
                          radius=.4, interface_layer=1)
    if already_refined:
        coarse, _ = refine_tetrahedra(coarse)
    fine, parents = refine_tetrahedra(coarse)
    a, b = prepare_mesh(coarse, 2), prepare_mesh(fine, 2)
    assert len(fine.cells) == 8*len(coarse.cells)
    np.testing.assert_array_equal(fine.physical_tags, coarse.physical_tags[parents])
    np.testing.assert_allclose(b.geometry.volumes*8, a.geometry.volumes[parents], rtol=1e-12)
    # Every child stays inside its parent's affine tetrahedron.
    delta = fine.vertices[fine.cells]-coarse.vertices[coarse.cells[parents, 0]][:, None]
    unit = np.einsum("eij,enj->eni", a.geometry.inverse_jacobians[parents], delta)/2
    assert np.min(np.concatenate((1-unit.sum(axis=-1, keepdims=True), unit), axis=-1)) >= -1e-12
    mask = free_surface_mask(b, (10,))
    points, weights = surface_quadrature(b, mask)
    areas = []
    for prepared in (a, b):
        selected = free_surface_mask(prepared, (10,))
        _, area = surface_quadrature(prepared, selected)
        areas.append(area.sum())
        xyz = prepared.geometry.nodes
        state = np.zeros((*xyz.shape[:2], 27))
        state[..., :3] = np.stack((xyz[..., 0]**2+xyz[..., 1]*xyz[..., 2],
                                  xyz[..., 0]*xyz[..., 1], xyz[..., 2]**2-.2*xyz[..., 0]), axis=-1)
        state[..., 9:] = xyz[..., 2, None]*np.arange(18)
        rows, interpolation = receiver_interpolation(prepared, points)
        actual = np.einsum("ri,ric->rc", interpolation, state[rows])
        expected = np.stack((points[:, 0]**2+points[:, 1]*points[:, 2],
                             points[:, 0]*points[:, 1], points[:, 2]**2-.2*points[:, 0]), axis=-1)
        np.testing.assert_allclose(actual[:, :3], expected, atol=1e-11, rtol=1e-11)
        positions, trace = surface_trace(prepared, state, selected)
        assert trace.shape[-1] == 27 and np.max(np.abs(trace[..., 9:])) > 0
        xyz = positions.reshape(-1, 3)[display_triangles(prepared.reference, len(positions))]
        display_area = np.linalg.norm(np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0]), axis=-1).sum()/2
        np.testing.assert_allclose(display_area, area.sum(), rtol=1e-12)
    np.testing.assert_allclose(areas[0], areas[1], rtol=1e-12)


@pytest.mark.parametrize("times", [[0, .2, .2], [0, .23], [-.1, 0], [0, 1.1]])
def test_invalid_snapshot_times(times):
    with pytest.raises(ValueError, match="snapshot"):
        snapshot_steps(times, .1, 10)


def test_weighted_wavefield_norm_and_time_selection():
    np.testing.assert_array_equal(snapshot_steps([0, .2, 1], .1, 10), [0, 2, 10])
    ref = np.array([[[0., 0, 0], [0, 0, 0]], [[1, 0, 0], [0, 2, 0]]])
    actual = ref.copy()
    actual[1, 0, 0] += 1
    result = field_comparison(actual, ref, np.array([1., 3.]), np.array([0., 1.]))
    assert result["spacetime_relative_l2"] == pytest.approx(1/np.sqrt(13))
    assert result["per_time_relative_l2"][0] == 0
    with pytest.raises(ValueError, match="zero reference"):
        field_comparison(actual+1, ref*0, np.array([1., 3.]), np.array([0., 1.]))


def test_source_turn_on_envelope_and_smt_rejection():
    config = load_config(Path(__file__).parents[1] / "volcano.toml")
    mesh = volcano_mesh()
    prepared = prepare_mesh(mesh, 2)
    old = smooth_source(prepared, config)
    config = replace(config, source=replace(config.source, ramp_time=.3))
    source = smooth_source(prepared, config)
    np.testing.assert_array_equal(source(0), 0)
    np.testing.assert_allclose(source(.15), .5*old(.15), atol=1e-14)
    np.testing.assert_allclose(source(.4), old(.4), atol=1e-14)
    check_physical_cores([[0, 0], [0, 1], [1, 0]])
    with pytest.raises(RuntimeError, match="SMT"):
        check_physical_cores([[0, 0], [0, 0]])


@pytest.mark.parametrize("empty", [False, True])
def test_selected_faces_match_all_face_reference_for_orientations(empty):
    from dg_wave.mesh import TetraMesh

    xyz = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for permutation in permutations((1, 2, 3)):
        cell = np.array([[0, *permutation]])
        if np.linalg.det((xyz[cell[0, 1:]]-xyz[cell[0, 0]]).T) < 0:
            cell[0, [0, 1]] = cell[0, [1, 0]]
        prepared = prepare_mesh(TetraMesh(xyz, cell, [731]), 2)
        mask = np.array([[False, True, True, False]]) & (not empty)
        state = np.random.default_rng(42).normal(size=(1, 10, 27))
        expected = np.einsum("fij,ejc->efic", prepared.reference.trace, state)[mask]
        positions, host = surface_trace(prepared, state, mask)
        actual = np.asarray(surface_trace_kernel(prepared, mask)(state))
        assert actual.shape == expected.shape == (0 if empty else 2, 6, 27)
        np.testing.assert_array_equal(positions, prepared.geometry.face_nodes[mask])
        np.testing.assert_allclose(host, expected, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize(("host", "linux", "expected", "safe"), [
    (3000, 15000, 1000, True),
    (4000, 15000, 2500, False),
    (10000, 4000, 2500, False),
])
def test_next_run_checks_measured_memory_headroom_before_launch(monkeypatch, tmp_path, host, linux, expected, safe):
    from scripts import wavefield_next

    monkeypatch.setattr(wavefield_next, "OUTPUT", tmp_path)
    monkeypatch.setattr(wavefield_next, "memory_available", lambda: {
        "windows_available_mib": host, "wsl_available_mib": linux,
    })

    def launch(*args, **kwargs):
        raise LookupError("launch reached")

    monkeypatch.setattr(wavefield_next.subprocess, "Popen", launch)
    with pytest.raises(LookupError if safe else RuntimeError,
                       match="launch reached" if safe else "memory headroom"):
        wavefield_next.execute("veryfine", [0., 1.2], 60, expected)
