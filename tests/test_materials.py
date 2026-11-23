from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

from dg_wave.config import Material, load_config, parse_config, tomllib
from dg_wave.materials import assign_materials
from dg_wave.mesh import TetraMesh, read_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.physics import wave_speeds
from dg_wave.run import receiver_interpolation, validate_source


ROOT = Path(__file__).parents[1]


def test_gmsh_physical_tags_and_layer_coefficients():
    config = load_config(ROOT / "layered_box.toml")
    mesh = read_mesh(config.mesh)
    material = assign_materials(mesh, config)
    z = mesh.vertices[mesh.cells, 2]
    np.testing.assert_array_equal(mesh.physical_tags, np.where(z.mean(axis=1) < 500, 1, 2))
    assert np.all((z.max(axis=1) <= 500) | (z.min(axis=1) >= 500))
    cp, cs = wave_speeds(material)
    np.testing.assert_allclose(cp, np.where(mesh.physical_tags == 1, 4500, 3000))
    np.testing.assert_allclose(cs, np.where(mesh.physical_tags == 1, 2500, 1600))
    assert not mesh.physical_tags.flags.writeable


@pytest.mark.parametrize("problem", ["untagged", "missing", "unused", "invalid_tag", "inverted", "degenerate"])
def test_invalid_tagged_mesh_and_material_assignment(problem, tmp_path):
    config = load_config(ROOT / "layered_box.toml")
    mesh = read_mesh(config.mesh)
    if problem == "untagged":
        mesh = TetraMesh(mesh.vertices, mesh.cells)
    elif problem == "missing":
        config = replace(config, materials=config.materials[:1])
    elif problem == "unused":
        config = replace(config, materials=(*config.materials, (3, Material(1, 2, 1))))
    else:
        cells, vertices, tags = mesh.cells.copy(), mesh.vertices.copy(), mesh.physical_tags.copy()
        if problem == "inverted":
            cells[0, [0, 1]] = cells[0, [1, 0]]
        elif problem == "degenerate":
            vertices[cells[0, 3]] = vertices[cells[0, :3]].mean(axis=0)
        else:
            tags[0] = 0
        path = tmp_path / "invalid.msh"
        meshio.write_points_cells(
            path, vertices, [("tetra", cells)], file_format="gmsh22", binary=False,
            cell_data={"gmsh:physical": [tags], "gmsh:geometrical": [tags]},
        )
        with pytest.raises(ValueError):
            prepare_mesh(read_mesh(path), config.degree)
        return
    with pytest.raises(ValueError, match="physical tags|tags mismatch"):
        assign_materials(mesh, config)


@pytest.mark.parametrize(("key", "value"), [
    ("rho", 0), ("vp", float("inf")), ("vs", -1), ("vp", 1000),
    ("vp", 1e308), ("vs", True),
])
def test_invalid_regional_material_parameters(key, value):
    with (ROOT / "layered_box.toml").open("rb") as stream:
        data = tomllib.load(stream)
    data["materials"]["1"][key] = value
    with pytest.raises(ValueError, match="materials"):
        parse_config(data)


@pytest.mark.parametrize("kind", ["source", "receiver"])
def test_layered_source_receiver_locations_rejected(kind):
    config = load_config(ROOT / "layered_box.toml")
    prepared = prepare_mesh(read_mesh(config.mesh), config.degree)
    with pytest.raises(ValueError, match="outside"):
        if kind == "source":
            validate_source(prepared, replace(config, source=replace(config.source, position=(500, 500, 1001))))
        else:
            receiver_interpolation(prepared, [(500, -1, 500)])


@pytest.mark.parametrize("problem", ["free_periodic", "sponge_periodic", "sponge_free", "material_key"])
def test_invalid_layered_boundary_or_tag_settings(problem):
    with (ROOT / "layered_box.toml").open("rb") as stream:
        data = tomllib.load(stream)
    if problem == "free_periodic":
        data["boundary"]["types"][2] = "periodic"
    elif problem == "sponge_periodic":
        data["boundary"]["types"][0] = "periodic"
    elif problem == "sponge_free":
        data["sponge"]["faces"] = ["zmax"]
    else:
        data["materials"]["rock"] = data["materials"].pop("1")
    with pytest.raises(ValueError, match="boundary|sponge|materials"):
        parse_config(data)
