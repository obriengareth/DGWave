import json

import numpy as np
import pytest

from dg_wave.rainier_layers import (
    INTERFACES_M, INTERVALS_TOP_DOWN, effective_materials, interface_report, layered_terrain_mesh, material_inputs,
)
from dg_wave.terrain import quality_report, surface_stations, terrain_mesh
from dg_wave.reference import FACE_VERTICES


@pytest.mark.parametrize("nonplanar", [False, True])
def test_six_material_regions_have_exact_conforming_depth_interfaces(nonplanar):
    top = np.full((5, 5), 3000.)
    if nonplanar:
        top += np.random.default_rng(42).uniform(-200, 200, top.shape)
    mesh = layered_terrain_mesh(top, width=2000.)
    original = terrain_mesh(top, width=2000.)
    np.testing.assert_array_equal(mesh.cells, original.cells)
    np.testing.assert_array_equal(mesh.element_ids, original.element_ids)
    np.testing.assert_array_equal(mesh.boundary_tags, original.boundary_tags)
    np.testing.assert_array_equal(mesh.vertices[:, :2], original.vertices[:, :2])
    np.testing.assert_array_equal(mesh.vertices[::11, 2], original.vertices[::11, 2])
    np.testing.assert_array_equal(mesh.vertices[10::11, 2], original.vertices[10::11, 2])
    np.testing.assert_array_equal(np.bincount(mesh.physical_tags)[1:], np.array(INTERVALS_TOP_DOWN)*4*4*6)
    report, faces, pairs = interface_report(mesh, top, 2000.)
    assert [r["depth_below_faceted_terrain_m"] for r in report] == list(INTERFACES_M)
    assert len(faces) == 5*32 and pairs.shape == (160, 2)
    assert quality_report(mesh)["min_volume_m3"] > 0
    before = surface_stations(original, inset=100., seed=42)
    after = surface_stations(mesh, inset=100., seed=42)
    assert before == after


def test_layered_mesh_rejects_a_pinched_basement():
    with pytest.raises(ValueError, match="deepest material interface"):
        layered_terrain_mesh([[0, 0], [0, 4500]], height=6000.)


def test_adopted_elastic_and_bulk_loss_consistency_without_q_fitting():
    table = effective_materials()
    for material in table:
        rho, k, mu = (material[name] for name in ("rho_kg_m3", "bulk_modulus_Pa", "shear_modulus_Pa"))
        assert k > 0 and mu > 0 and 0 < material["poisson_ratio"] < .5
        assert np.sqrt((k+4*mu/3)/rho) == pytest.approx(material["vp_m_s"], rel=1e-14)
        assert np.sqrt(mu/rho) == pytest.approx(material["vs_m_s"], rel=1e-14)
        p_loss = (k+4*mu/3)/material["qp"]
        shear_loss = 4*mu/(3*material["qs"])
        assert material["necessary_inverse_bulk_q"] == pytest.approx((p_loss-shear_loss)/k)
        assert material["necessary_inverse_bulk_q"] > 0


def test_velocity_view_uses_owning_material_for_real_full_and_cut_faces(tmp_path):
    from scripts.rainier_view import write_view

    mesh = layered_terrain_mesh(np.full((3, 3), 3000.), width=2000.)
    write_view(tmp_path, mesh, {"materials": effective_materials()})
    text = (tmp_path / "view.html").read_text()
    data, _ = json.JSONDecoder().raw_decode(text.split("const data=", 1)[1])
    owners = {}
    for cell, tag in zip(mesh.cells, mesh.physical_tags, strict=True):
        for face in cell[FACE_VERTICES]:
            owners.setdefault(tuple(sorted(face)), set()).add(int(tag))
    for faces, materials in (("full", "fullMaterials"), ("cut", "cutMaterials")):
        assert set(data[materials]) == set(range(1, 7))
        for face, tag in zip(data[faces], data[materials], strict=True):
            assert owners[tuple(sorted(face))] == {tag}


def test_adopted_material_fragment_requires_explicit_attenuation_band():
    from pathlib import Path
    from dg_wave.config import parse_config, tomllib

    with (Path(__file__).parents[1]/"attenuating_box.toml").open("rb") as stream:
        data = tomllib.load(stream)
    fragment = material_inputs()
    assert set(fragment["attenuation"]) == {"materials"}
    band = {key: data["attenuation"][key] for key in ("band", "reference_frequency", "mechanisms")}
    data.update(fragment)
    with pytest.raises(ValueError, match="attenuation must contain"):
        parse_config(data)
    data["attenuation"].update(band)
    config = parse_config(data)
    assert config.attenuation.qp is None and config.attenuation.qs is None
    assert config.attenuation.material_q == tuple((row["tag"], row["qp"], row["qs"]) for row in effective_materials())
