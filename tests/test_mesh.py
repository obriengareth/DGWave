import meshio
import numpy as np
import pytest

from dg_wave.mesh import TetraMesh, box_mesh, read_mesh


def test_box_fixture_and_stable_ids(tmp_path):
    mesh = box_mesh()
    assert mesh.cells.shape == (6, 4)
    assert len(set(mesh.element_ids)) == 6
    path = tmp_path / "box.vtu"
    meshio.write_points_cells(
        path, mesh.vertices, [("tetra", mesh.cells)],
        cell_data={"element_id": [mesh.element_ids]},
    )
    restored = read_mesh(path)
    np.testing.assert_array_equal(restored.vertices, mesh.vertices)
    np.testing.assert_array_equal(restored.cells, mesh.cells)
    np.testing.assert_array_equal(restored.element_ids, mesh.element_ids)
    order = np.array([4, 0, 5, 1, 3, 2])
    reordered = TetraMesh(mesh.vertices, mesh.cells[order], mesh.element_ids[order])
    np.testing.assert_array_equal(reordered.element_ids, mesh.element_ids[order])
    assert not reordered.cells.flags.writeable


@pytest.mark.parametrize("defect", [
    "duplicate_id", "fractional_cell", "repeated_vertex", "out_of_range", "duplicate_cell",
])
def test_invalid_mesh(defect):
    mesh = box_mesh()
    cells = mesh.cells.copy()
    ids = mesh.element_ids.copy()
    if defect == "duplicate_id":
        ids[1] = ids[0]
    elif defect == "fractional_cell":
        cells = cells.astype(float) + 0.1
    elif defect == "repeated_vertex":
        cells[0, 1] = cells[0, 0]
    elif defect == "out_of_range":
        cells[0, 1] = len(mesh.vertices)
    else:
        cells[1] = cells[0]
    with pytest.raises(ValueError):
        TetraMesh(mesh.vertices, cells, ids)
