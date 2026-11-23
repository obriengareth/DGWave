"""Build the confirmed-depth/Vp effective layer mesh without running waves."""

from scripts.rainier_model import JobGuard, ROOT

import copy
import argparse
import hashlib
import json
from pathlib import Path
import resource
import shutil
import time
from types import SimpleNamespace

import meshio
import numpy as np

from dg_wave.config import Material
from dg_wave.materials import assign_materials
from dg_wave.mesh import TetraMesh
from dg_wave.rainier_layers import (
    INTERFACES_M, INTERVALS_TOP_DOWN, effective_materials, interface_report, layered_terrain_mesh, material_inputs,
)
from dg_wave.reference import FACE_VERTICES
from dg_wave.terrain import quality_report, surface_stations
from scripts.rainier_view import write_view

ORIGINAL = ROOT / "output" / "rainier_model"
OUTPUT = ROOT / "output" / "rainier_layered_model"


def hashes(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.iterdir()) if path.is_file()}


def adopt_material_metadata(metadata, table):
    metadata.pop("proposed_materials", None)
    metadata["materials"] = table
    metadata["simulation_ready"] = False
    metadata["layer_approval"] = {
        "user_confirmed": "six depth intervals,Vp,density,Vs,Qp/Qs",
        "still_unselected": "source, attenuation band/reference/mechanism count, degree/timestep and run/output settings",
        "observational_status": "agreed engineering assumptions, not measured/calibrated Rainier profiles or mapped contacts",
    }
    metadata["attenuation_status"] = {
        "fitted": False, "source_band_and_reference_frequency": None,
        "necessary_passivity_check": "positive K/mu and positive inverse bulkQ=(rho*Vp^2/Qp-4rho*Vs^2/(3Qs))/K for every region",
        "limitation": "necessary positive bulk/shear dissipation only; no finite-band passive Maxwell fit or Q-accuracy claim before band/reference is agreed",
        "solver_configuration": "per-tag Q inputs implemented in attenuation.materials, using the shared Maxwell kernels; material_inputs.json is a partial config, not a runnable simulation",
        "elastic_convention": "Vp/Vs define elastic storage-derived K/mu; no claim of calibrated finite-Q phase velocities",
    }
    metadata["remaining_choices"] = [
        "agree source position/type/band/reference-frequency/duration and run/output/checkpoint settings",
        "select mechanism count and fit/verify passive layer attenuation over the agreed band",
        "choose degree/timestep and verify full-job RAM by bounded pilot before any Rainier propagation",
    ]


def write_material_metadata(metadata):
    (OUTPUT / "material_inputs.json").write_text(json.dumps(material_inputs(), indent=2))
    (OUTPUT / "materials.json").write_text(json.dumps({
        "approval": metadata["layer_approval"], "materials": metadata["materials"],
        "attenuation_status": metadata["attenuation_status"],
    }, indent=2))


def adopt_existing():
    protected = ("rainier.msh", "mesh.npz", "interfaces.npz", "stations.csv", "receivers.json")
    before = hashes(OUTPUT)
    preserved = {name: before[name] for name in protected}
    neutral = hashes(ORIGINAL)
    metadata = json.loads((OUTPUT / "model.json").read_text())
    with JobGuard() as guard:
        table = effective_materials()
        adopt_material_metadata(metadata, table)
        with np.load(OUTPUT / "element_materials.npz", allow_pickle=False) as arrays:
            values = {name: arrays[name] for name in arrays.files}
        lookup = values["material_tags"]-1
        for name in ("qp", "qs"):
            adopted = np.asarray([row[name] for row in table])[lookup]
            np.testing.assert_array_equal(adopted, values.pop(name+"_proposed", values.get(name)))
            values[name] = adopted
        np.savez_compressed(OUTPUT / "element_materials.npz", **values)
        with np.load(OUTPUT / "mesh.npz", allow_pickle=False) as arrays:
            mesh = TetraMesh(arrays["vertices"], arrays["cells"], arrays["element_ids"],
                             arrays["material_tags"], arrays["boundary_tags"])
        write_material_metadata(metadata)
        write_view(OUTPUT, mesh, metadata)
    metadata["material_adoption"] = {
        "mesh_interfaces_and_stations_sha256": preserved,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "minimum_sampled_windows_available_bytes": guard.minimum,
        "q_fit_or_propagation_performed": False,
    }
    after = hashes(OUTPUT)
    if any(after[name] != digest for name, digest in preserved.items()) or hashes(ORIGINAL) != neutral:
        raise ValueError("material adoption changed protected geometry/stations or neutral model")
    (OUTPUT / "model.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["material_adoption"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adopt-existing", action="store_true", help="Update material approval and Q inputs without remeshing")
    if parser.parse_args().adopt_existing:
        adopt_existing()
        return
    started = time.perf_counter()
    if OUTPUT.exists():
        raise ValueError("refusing to overwrite an existing layered Rainier model")
    preserved = hashes(ORIGINAL)
    original = json.loads((ORIGINAL / "model.json").read_text())
    with JobGuard() as guard:
        with np.load(ORIGINAL / "terrain_samples.npz", allow_pickle=False) as data:
            top = data["elevation_m"].copy()
        width, height = original["dimensions"]["width_x_m"], original["dimensions"]["total_height_m"]
        mesh = layered_terrain_mesh(top, width, height)
        quality = quality_report(mesh)
        reports, triangles, pairs = interface_report(mesh, top, width)
        if quality["elements"] != 150000 or quality["vertices"] != 28611:
            raise ValueError("layered model must retain exactly150000 cells and28611 vertices")
        with np.load(ORIGINAL / "mesh.npz", allow_pickle=False) as old:
            for name, actual in (("cells", mesh.cells), ("element_ids", mesh.element_ids),
                                 ("boundary_tags", mesh.boundary_tags)):
                np.testing.assert_array_equal(actual, old[name])
            np.testing.assert_array_equal(mesh.vertices[:, :2], old["vertices"][:, :2])
            np.testing.assert_array_equal(mesh.vertices[::11, 2], old["vertices"][::11, 2])
            np.testing.assert_array_equal(mesh.vertices[10::11, 2], old["vertices"][10::11, 2])
            moved = int(np.count_nonzero(mesh.vertices[:, 2] != old["vertices"][:, 2]))
            maximum_move = float(np.max(np.abs(mesh.vertices[:, 2]-old["vertices"][:, 2])))
        generated = surface_stations(mesh, seed=original["stations"]["seed"],
                                     inset=original["stations"]["edge_inset_m"])
        for actual, expected in zip(generated, original["stations"]["items"], strict=True):
            if actual["id"] != expected["id"] or actual["element_id"] != expected["element_id"] or actual["face"] != expected["face"]:
                raise ValueError("surface station identity/owning face changed")
            np.testing.assert_array_equal(actual["xyz_m"], expected["xyz_m"])
            np.testing.assert_array_equal(actual["triangle_weights"], expected["triangle_weights"])
        table = effective_materials()
        configured = SimpleNamespace(material=None, materials=tuple(
            (row["tag"], Material(row["rho_kg_m3"], row["bulk_modulus_Pa"], row["shear_modulus_Pa"]))
            for row in table
        ))
        elastic = assign_materials(mesh, configured)
        lookup = mesh.physical_tags-1
        material_arrays = {
            "element_ids": mesh.element_ids, "material_tags": mesh.physical_tags,
            "density_kg_m3": elastic.density, "bulk_modulus_Pa": elastic.bulk_modulus,
            "shear_modulus_Pa": elastic.shear_modulus,
            **{name: np.asarray([row[name] for row in table])[lookup]
               for name in ("vp_m_s", "vs_m_s", "qp", "qs")},
        }
        np.testing.assert_allclose(
            np.sqrt((elastic.bulk_modulus+4*elastic.shear_modulus/3)/elastic.density),
            material_arrays["vp_m_s"], atol=1e-10, rtol=1e-14,
        )
        np.testing.assert_allclose(np.sqrt(elastic.shear_modulus/elastic.density),
                                   material_arrays["vs_m_s"], atol=1e-10, rtol=1e-14)
        region_counts = {str(row["tag"]): int(np.count_nonzero(mesh.physical_tags == row["tag"])) for row in table}
        OUTPUT.mkdir()
        rows, faces = np.nonzero(mesh.boundary_tags)
        boundary_tags = mesh.boundary_tags[rows, faces]
        meshio.write_points_cells(
            OUTPUT / "rainier.msh", mesh.vertices,
            [("tetra", mesh.cells), ("triangle", mesh.cells[rows[:, None], FACE_VERTICES[faces]])],
            file_format="gmsh22", binary=False,
            cell_data={"gmsh:physical": [mesh.physical_tags, boundary_tags],
                       "gmsh:geometrical": [mesh.physical_tags, boundary_tags]},
            field_data={**{f"effective_region_{row['tag']}": [row["tag"], 3] for row in table},
                        "terrain": [10, 2], "outer": [20, 2]},
        )
        np.savez_compressed(OUTPUT / "mesh.npz", vertices=mesh.vertices, cells=mesh.cells,
                            element_ids=mesh.element_ids, material_tags=mesh.physical_tags,
                            boundary_tags=mesh.boundary_tags)
        np.savez_compressed(OUTPUT / "interfaces.npz", triangles=triangles, adjacent_material_tags=pairs)
        np.savez_compressed(OUTPUT / "element_materials.npz", **material_arrays)
        for name in ("stations.csv", "receivers.json"):
            shutil.copyfile(ORIGINAL / name, OUTPUT / name)
        metadata = copy.deepcopy(original)
        metadata.update(
            model="Mount Rainier six-layer simplified effective model",
            simulation_ready=False,
            geological_interpretation={
                "boundary_definition": "vertical depth below the SAME piecewise-planar DSM-derived simulation terrain",
                "interface_depths_m": list(INTERFACES_M),
                "limitations": [
                    "All five interfaces are terrain-following fixed-depth surfaces, including basement; this is intentional geometric simplification, not mapped stratigraphy.",
                    "Rainier is an andesitic edifice over complex older volcanic/sedimentary/intrusive rocks, not six laterally uniform depth-only units.",
                    "The0.5/0.875/1.25/1.625/2km depths and2.5–5km/s velocities are agreed modelling assumptions, not values inferred from the cited papers.",
                    "The same layers extend across the entire25km square, including valleys and pre-edifice rock; no lateral mapped geology, hydrothermal alteration or ice/bedrock separation is represented.",
                    "The deepest effective unit is only208.095m thick at the shallowest terrain column and4000m thick beneath the retained summit.",
                    "The published5–18km low-velocity anomaly is NOT represented as a magma chamber or used to assign these layers.",
                ],
                "context_sources": [
                    {"url": "https://pubs.usgs.gov/publication/pp444",
                     "citation": "Fiske,Hopson,Waters1963,USGS Professional Paper444",
                     "supports": "andesitic edifice over older, structurally complex volcanic/sedimentary/intrusive rock; not the proposed depth/Vp values"},
                    {"url": "https://pubs.usgs.gov/publication/70021644",
                     "citation": "Moran,Lees,Malone1999,doi10.1029/1999JB900036",
                     "supports": "upper-crust velocity heterogeneity/geology correlation and deeper5–18km low-velocity anomaly; not these contacts or properties"},
                ],
            },
            mesh={**quality, "structured_divisions": [50, 50, 10],
                  "vertical_intervals_top_down": list(INTERVALS_TOP_DOWN),
                  "cell_counts_by_material_tag": region_counts,
                  "material_tags": {str(row["tag"]): row["name"] for row in table},
                  "boundary_tags": original["mesh"]["boundary_tags"],
                  "interface_faces": reports, "topology_and_stable_element_ids_unchanged": True,
                  "moved_interior_vertices": moved, "maximum_vertical_vertex_remap_m": maximum_move,
                  "external_surface_and_base_vertices_bitwise_preserved": True},
            original_neutral_model={"relative_path": "../rainier_model", "file_sha256": preserved},
        )
        adopt_material_metadata(metadata, table)
        write_material_metadata(metadata)
        write_view(OUTPUT, mesh, metadata)
    metadata["layer_setup_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    metadata["layer_setup_peak_rss_method"] = "Linux single process RUSAGE_SELF ru_maxrss KiB*1024, geometry/material/view setup only"
    metadata["layer_setup_minimum_sampled_windows_available_bytes"] = guard.minimum
    metadata["layer_setup_wall_seconds"] = time.perf_counter()-started
    metadata["geometry_peak_rss_bytes"] = original["geometry_peak_rss_bytes"]
    if hashes(ORIGINAL) != preserved:
        raise ValueError("the original neutral model/provenance/station files changed")
    (OUTPUT / "model.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
