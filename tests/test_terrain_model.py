import numpy as np
import pytest

from dg_wave.reference import FACE_VERTICES
from dg_wave.terrain import faceted_elevation, quality_report, surface_stations, terrain_mesh


@pytest.mark.parametrize("slope", [0., .2])
def test_terrain_geometry_tags_and_surface_stations(slope):
    x, y = np.meshgrid(np.linspace(-1000, 1000, 5), np.linspace(-1000, 1000, 5), indexing="ij")
    top = 1200+slope*x+.1*y
    mesh = terrain_mesh(top, width=2000., height=3000., layers=3)
    report = quality_report(mesh)
    assert report["elements"] == 4*4*3*6
    assert report["min_volume_m3"] > 0 and report["terrain_faces"] == 32
    assert mesh.vertices[:, 2].max()-mesh.vertices[:, 2].min() == pytest.approx(3000.)
    stations = surface_stations(mesh, inset=100., seed=47)
    assert stations == surface_stations(mesh, inset=100., seed=47)
    assert len({s["id"] for s in stations}) == len({tuple(s["xyz_m"]) for s in stations}) == 10
    for station in stations:
        xyz = np.asarray(station["xyz_m"])
        assert np.all(np.abs(xyz[:2]) <= 900.)
        assert xyz[2] == pytest.approx(1200+slope*xyz[0]+.1*xyz[1])
        row, = np.flatnonzero(mesh.element_ids == station["element_id"])
        assert mesh.boundary_tags[row, station["face"]] == 10
        triangle = mesh.vertices[mesh.cells[row, FACE_VERTICES[station["face"]]]]
        np.testing.assert_allclose(np.asarray(station["triangle_weights"]) @ triangle, xyz)


@pytest.mark.parametrize("top,height", [
    ([[1., float("nan")], [2., 3.]], 5),
    ([[0., 1.], [2., 3.]], 2),
    ([1., 2.], 5),
])
def test_invalid_terrain_rejected(top, height):
    with pytest.raises(ValueError, match="terrain|bottom"):
        terrain_mesh(top, height=height)


def test_nonplanar_facet_sampling_matches_actual_station_triangles():
    top = 1000+np.random.default_rng(16).uniform(0, 300, (5, 5))
    mesh = terrain_mesh(top, width=2000., height=3000., layers=3)
    stations = surface_stations(mesh, seed=19, inset=100.)
    xyz = np.asarray([s["xyz_m"] for s in stations])
    np.testing.assert_allclose(faceted_elevation(top, xyz[:, :2], 2000.), xyz[:, 2], atol=1e-9)
