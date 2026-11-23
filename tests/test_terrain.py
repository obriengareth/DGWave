import meshio
import numpy as np
import pytest

from dg_wave.config import Material, parse_config, tomllib
from dg_wave.mesh import TetraMesh, read_mesh
from dg_wave.operators import prepare_mesh
from dg_wave.physics import interface_states, stress_matrix
from dg_wave.solver import ElasticSolver
from dg_wave.volcano import volcano_mesh, write_volcano


def test_sloping_terrain_orientation_tags_and_zero_traction(tmp_path):
    path = tmp_path / "terrain.msh"
    original = write_volcano(path, divisions=(2, 2, 2), interface_layer=1)
    mesh = read_mesh(path)
    np.testing.assert_array_equal(mesh.boundary_tags, original.boundary_tags)
    prepared = prepare_mesh(mesh, 2)
    solver = ElasticSolver(prepared, Material(1, 8/3, 1), ("absorbing",)*3,
                           free_surface=True, free_surface_tags=(10,))
    mask = solver.free_faces
    normals = prepared.geometry.normals[mask]
    assert np.all(normals[:, 2] > 0) and np.any(np.linalg.norm(normals[:, :2], axis=-1) > 0.1)
    assert np.all(prepared.geometry.volumes > 0)
    # Geometry normals point away from each cell, including sloping terrain.
    from dg_wave.reference import FACE_VERTICES
    centres = mesh.vertices[mesh.cells[:, FACE_VERTICES]].mean(axis=2)
    cell_centres = mesh.vertices[mesh.cells].mean(axis=1)
    assert np.all(np.sum(prepared.geometry.normals*(centres-cell_centres[:, None]), axis=-1) > 0)
    left = np.random.default_rng(7).normal(size=(len(normals), 9))
    _, traction = interface_states(left[:, None], left[:, None]*0, normals[:, None],
                                  Material(1, 8/3, 1), Material(1, 8/3, 1),
                                  free_surface=np.ones(len(normals), dtype=bool))
    np.testing.assert_allclose(traction, 0, atol=1e-14)
    assert np.any(np.linalg.norm(np.einsum("fij,fj->fi", stress_matrix(left), normals), axis=-1) > 0)
    with pytest.raises(ValueError, match="missing"):
        ElasticSolver(prepared, Material(1, 8/3, 1), ("absorbing",)*3,
                      free_surface=True, free_surface_tags=(99,))


@pytest.mark.parametrize("problem", ["interior_tag", "inverted", "triangle", "parameters"])
def test_invalid_terrain_geometry_and_tags(tmp_path, problem):
    mesh = volcano_mesh(divisions=(2, 2, 2), interface_layer=1)
    if problem == "parameters":
        with pytest.raises(ValueError, match="terrain"):
            volcano_mesh(radius=0)
    elif problem == "interior_tag":
        tags = np.ones_like(mesh.boundary_tags)
        with pytest.raises(ValueError, match="exterior"):
            TetraMesh(mesh.vertices, mesh.cells, boundary_tags=tags)
    elif problem == "inverted":
        cells = mesh.cells.copy()
        cells[0, [0, 1]] = cells[0, [1, 0]]
        with pytest.raises(ValueError, match="inverted"):
            prepare_mesh(TetraMesh(mesh.vertices, cells), 1)
    else:
        path = tmp_path / "invalid.msh"
        meshio.write_points_cells(
            path, mesh.vertices, [("tetra", mesh.cells), ("triangle", [[0, 1, len(mesh.vertices)-1]])],
            file_format="gmsh22", binary=False,
            cell_data={"gmsh:physical": [mesh.physical_tags, [10]],
                       "gmsh:geometrical": [mesh.physical_tags, [10]]},
        )
        with pytest.raises(ValueError, match="exterior"):
            read_mesh(path)


@pytest.mark.parametrize(("section", "key", "value"), [
    ("boundary", "free_tags", [10, 10]),
    ("source", "band", [3, 0.5]),
    ("source", "band", [0.1, 4]),
])
def test_invalid_terrain_boundary_and_source_config(section, key, value):
    from pathlib import Path
    with (Path(__file__).parents[1] / "volcano.toml").open("rb") as stream:
        data = tomllib.load(stream)
    data[section][key] = value
    with pytest.raises(ValueError):
        parse_config(data)
