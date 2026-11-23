"""MPC terrain-only Rainier geometry; no geological properties or wave solve."""

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "GDAL_NUM_THREADS"):
    os.environ[variable] = "1"

import meshio
import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.windows import Window, transform as window_transform
from rasterio.warp import transform_bounds

from dg_wave.reference import FACE_VERTICES
from dg_wave.terrain import faceted_elevation, quality_report, surface_stations, terrain_mesh

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "output" / "rainier_model"
STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
WIDTH = 25000.
HEIGHT = 6000.
ANCHOR = (-121.760, 46.853)
DATA_PAGE = "https://dataspace.copernicus.eu/explore-data/data-collections/copernicus-contributing-missions/collections-description/COP-DEM"
ATTRIBUTION = ("produced using Copernicus WorldDEM-30 \u00a9 DLR e.V. 2010-2014 and "
               "\u00a9 Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS "
               "by the European Union and ESA; all rights reserved")


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def host_available():
    result = subprocess.run(
        ["/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe", "-NoProfile",
         "-NonInteractive", "-Command",
         "[double](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1024"],
        capture_output=True, text=True, timeout=20, check=True,
    )
    return float(result.stdout.strip().lstrip("\ufeff"))


class JobGuard:
    """Conservative2GB geometry budget beneath the user's15GB total-job ceiling."""
    def __init__(self):
        self.stop = threading.Event()
        self.error = None
        self.minimum = host_available()
        self.budget = 2_000_000_000
        if self.minimum < self.budget+2_000_000_000:
            raise RuntimeError("geometry job requires2GB estimated demand plus2GB free host reserve")

    def monitor(self):
        try:
            while not self.stop.wait(5):
                self.minimum = min(self.minimum, host_available())
                peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
                if self.minimum < 2_000_000_000 or peak > self.budget:
                    raise RuntimeError("Rainier geometry RAM guard stopped the job")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            self.error = error
            os.kill(os.getpid(), signal.SIGUSR1)

    def __enter__(self):
        def fail(signum, frame):
            raise RuntimeError(f"Rainier resource monitor: {self.error}")
        self.previous = signal.signal(signal.SIGUSR1, fail)
        self.thread = threading.Thread(target=self.monitor, name="rainier-memory-guard")
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        signal.signal(signal.SIGUSR1, self.previous)
        self.minimum = min(self.minimum, host_available())
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > self.budget:
            raise RuntimeError("geometry stage exceeded its conservative2GB memory budget")


def search(collection, bbox):
    query = urllib.parse.urlencode({"collections": collection, "bbox": ",".join(map(str, bbox)), "limit": 100})
    result = fetch(STAC+"/search?"+query)
    if any(link["rel"] == "next" for link in result["links"]):
        raise ValueError("coverage search requires pagination; refusing incomplete selection")
    return result["features"]


def bilinear(data, transform, lon, lat):
    cols, rows = ~transform * (np.asarray(lon), np.asarray(lat))
    cols, rows = cols-.5, rows-.5
    c, r = np.floor(cols).astype(int), np.floor(rows).astype(int)
    if np.any(c < 0) or np.any(r < 0) or np.any(c+1 >= data.shape[1]) or np.any(r+1 >= data.shape[0]):
        raise ValueError("requested point falls outside bounded DEM crop")
    u, v = cols-c, rows-r
    return ((1-u)*(1-v)*data[r, c]+u*(1-v)*data[r, c+1]
            +(1-u)*v*data[r+1, c]+u*v*data[r+1, c+1])


def main():
    started = time.perf_counter()
    if OUTPUT.exists():
        raise ValueError("Rainier output directory already exists; refusing to overwrite model evidence")
    OUTPUT.mkdir(parents=True)
    forward = Transformer.from_crs(4326, 32610, always_xy=True)
    backward = Transformer.from_crs(32610, 4326, always_xy=True)
    center = np.array(forward.transform(*ANCHOR))
    projected_bounds = [*(center-WIDTH/2), *(center+WIDTH/2)]
    bbox = transform_bounds(32610, 4326, *projected_bounds, densify_pts=21)
    with JobGuard() as guard:
        dtm = search("3dep-lidar-dtm", bbox)
        dsm = search("cop-dem-glo-30", bbox)
        if any(item["bbox"][0] <= ANCHOR[0] <= item["bbox"][2]
               and item["bbox"][1] <= ANCHOR[1] <= item["bbox"][3] for item in dtm):
            raise ValueError("3DEP coverage changed: re-evaluate preferred complete DTM before fallback")
        if len(dsm) != 1 or not (dsm[0]["bbox"][0] < bbox[0] < bbox[2] < dsm[0]["bbox"][2]
                                and dsm[0]["bbox"][1] < bbox[1] < bbox[3] < dsm[0]["bbox"][3]):
            raise ValueError("one complete Copernicus tile required; no silent mosaic/datum mixing")
        item = dsm[0]
        asset = item["assets"]["data"]["href"]
        signed = fetch("https://planetarycomputer.microsoft.com/api/sas/v1/sign?href="
                       +urllib.parse.quote(asset, safe=""))["href"]
        with rasterio.Env(GDAL_CACHEMAX=32*1024**2, CPL_VSIL_CURL_CACHE_SIZE=8*1024**2,
                          GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT=40):
            with rasterio.open(signed) as source:
                corners = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
                cols, rows = ~source.transform * corners.T
                c0, r0 = np.floor([cols.min()-3, rows.min()-3]).astype(int)
                c1, r1 = np.ceil([cols.max()+3, rows.max()+3]).astype(int)
                window = Window(int(c0), int(r0), int(c1-c0), int(r1-r0))
                if window.width*window.height > 4_000_000:
                    raise ValueError("DEM window exceeds bounded4M-pixel geometry budget")
                raster = source.read(1, window=window, masked=True)
                if np.any(np.ma.getmaskarray(raster)) or not np.isfinite(raster).all():
                    raise ValueError("DEM crop contains nodata/nonfinite elevations; no silent filling")
                data = np.asarray(raster)
                transform = window_transform(window, source.transform)
                raster_info = {
                    "crs": str(source.crs), "nodata": source.nodata, "full_shape": source.shape,
                    "window_col_row_width_height": list(window.flatten()), "crop_shape": data.shape,
                    "crop_uncompressed_bytes": data.nbytes, "source_tags": source.tags(),
                    "full_source_tile_downloaded": False, "cache_bytes": 40*1024**2,
                }
                with rasterio.open(OUTPUT / "dem_crop.tif", "w", driver="GTiff", width=data.shape[1],
                                   height=data.shape[0], count=1, dtype="float32", crs=source.crs,
                                   transform=transform, compress="deflate") as target:
                    target.write(data, 1)
        axis = np.linspace(-WIDTH/2, WIDTH/2, 51)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        lon, lat = backward.transform(center[0]+x, center[1]+y)
        top = bilinear(data, transform, lon, lat)
        mesh = terrain_mesh(top, WIDTH, HEIGHT, 10)
        quality = quality_report(mesh)
        if quality["elements"] != 150000 or quality["vertices"] != 28611:
            raise ValueError("expected exactly50x50x10 structured cells split into150000 tetrahedra")
        np.testing.assert_allclose(np.ptp(mesh.vertices[:, :2], axis=0), [WIDTH, WIDTH], atol=1e-8)
        if abs(np.ptp(mesh.vertices[:, 2])-HEIGHT) > 1e-8:
            raise ValueError("model vertical extent is not6000m")
        stations = surface_stations(mesh, seed=20261120, inset=500.)
        for station in stations:
            xy = np.asarray(station["xyz_m"][:2])+center
            station["utm10n_xy_m"] = xy.tolist()
            station["lon_lat_degrees"] = list(backward.transform(*xy))
        elements, faces = np.nonzero(mesh.boundary_tags)
        tags = mesh.boundary_tags[elements, faces]
        meshio.write_points_cells(
            OUTPUT / "rainier.msh", mesh.vertices,
            [("tetra", mesh.cells), ("triangle", mesh.cells[elements[:, None], FACE_VERTICES[faces]])],
            file_format="gmsh22", binary=False,
            cell_data={"gmsh:physical": [mesh.physical_tags, tags], "gmsh:geometrical": [mesh.physical_tags, tags]},
            field_data={"unassigned_geometry_only": [1, 3], "terrain": [10, 2], "outer": [20, 2]},
        )
        test_axis = np.linspace(-WIDTH/2, WIDTH/2, 201)
        xx, yy = np.meshgrid(test_axis, test_axis, indexing="ij")
        ll, pp = backward.transform(center[0]+xx, center[1]+yy)
        native = bilinear(data, transform, ll, pp)
        modeled = faceted_elevation(top, np.column_stack((xx.ravel(), yy.ravel())), WIDTH).reshape(xx.shape)
        difference = modeled-native
        np.savez_compressed(OUTPUT / "terrain_samples.npz", x_m=x, y_m=y, elevation_m=top,
                            validation_x_m=xx, validation_y_m=yy, dem_validation_elevation_m=native,
                            model_validation_elevation_m=modeled)
        top_index = np.unravel_index(np.argmax(top), top.shape)
        edge = np.concatenate((top[0], top[-1], top[:, 0], top[:, -1]))
        bottom = float(mesh.vertices[:, 2].min())
        np.savez_compressed(OUTPUT / "mesh.npz", vertices=mesh.vertices, cells=mesh.cells,
                            element_ids=mesh.element_ids, material_tags=mesh.physical_tags,
                            boundary_tags=mesh.boundary_tags)
        metadata = {
            "model": "Mount Rainier geometry only", "simulation_ready": False,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "summit_anchor": {"longitude": ANCHOR[0], "latitude": ANCHOR[1],
                              "source": "https://www.usgs.gov/volcanoes/mount-rainier",
                              "meaning": "published approximate summit coordinate, not a resurveyed current maximum",
                              "published_nominal_elevation_m_not_used_for_base": 4392},
            "coordinates": {"horizontal_crs": "EPSG:32610 WGS84/UTM10N",
                            "mesh_xy": "local east/north metres relative to summit anchor",
                            "utm_origin_m": center.tolist(), "utm_bounds_m": projected_bounds,
                            "wgs84_search_bbox": bbox, "vertical_crs": "EPSG:3855 EGM2008 orthometric metres",
                            "vertical_transformation": "none; source EGM2008 retained, not NAVD88 or ellipsoidal height"},
            "dimensions": {"width_x_m": WIDTH, "width_y_m": WIDTH, "total_height_m": HEIGHT,
                           "mesh_highest_elevation_m": float(top.max()), "flat_base_elevation_m": bottom,
                           "mesh_highest_xy_m": [float(x[top_index]), float(y[top_index])],
                           "retained_terrain_min_max_m": [float(top.min()), float(top.max())],
                           "edge_terrain_min_max_m": [float(edge.min()), float(edge.max())],
                           "local_depth_min_max_m": [float(top.min()-bottom), HEIGHT],
                           "native_crop_max_m_not_retained_as_mesh_peak": float(data.max())},
            "dataset": {"collection": "cop-dem-glo-30", "item_id": item["id"], "unsigned_asset": asset,
                        "stac_properties": item["properties"], "stac_datetime_is_not_exact_acquisition": True,
                        "acquisition": "main TanDEM-X2011-2015; older infill possible; exact local acquisition dates not in MPC item",
                        "source_type": "DSM, not bare-earth DTM; includes vegetation/infrastructure and glacier/snow surface",
                        "glaciers": "effective terrain/ice surface only; no ice thickness, bedrock correction or separate ice medium",
                        "license": "Copernicus GLO30 free license; attribution/derived-product notice required",
                        "license_and_datum_source": DATA_PAGE, "attribution": ATTRIBUTION,
                        "doi": "https://doi.org/10.5270/ESA-c5d3d65", "raster": raster_info},
            "preferred_3dep_check": {
                "collection": "3dep-lidar-dtm", "summit_covered": False,
                "rejection": "only corner tiles, no central/summit coverage; do not combine NAVD88 DTM with EGM2008 DSM",
                "items": [{"id": t["id"], "bbox": t["bbox"],
                           "start": t["properties"].get("start_datetime"),
                           "end": t["properties"].get("end_datetime"),
                           "crs": t["properties"].get("proj:projjson", {}).get("name")} for t in dtm]},
            "terrain_sampling": {"mesh_spacing_m": 500, "mesh_nodes_xy": [51, 51],
                                 "method": "bilinear point samples of native cropped DSM at mesh vertices; affine triangles, no smoothing/vertical exaggeration",
                                 "fidelity_grid_spacing_m": 125, "fidelity_points": int(native.size),
                                 "model_minus_DSM_rms_m": float(np.sqrt(np.mean(difference**2))),
                                 "model_minus_DSM_max_abs_m": float(np.max(np.abs(difference))),
                                 "model_minus_DSM_abs_p95_m": float(np.quantile(np.abs(difference), .95)),
                                 "fidelity_caveat": "coarse500m faceted terrain is an approximation, not30m model resolution or geological accuracy"},
            "mesh": {**quality, "structured_divisions": [50, 50, 10], "material_region": "tag1 UNASSIGNED placeholder, not a lithology",
                     "geometric_levels": "10 terrain-following sigma intervals, NOT geological layers",
                     "boundary_tags": {"10": "terrain free-surface candidate", "20": "lateral/base absorbing candidate"}},
            "stations": {"count": 10, "seed": 20261120, "edge_inset_m": 500,
                         "placement": "uniform random XY, actual free-triangle barycentric Z; surface-station assumption",
                         "items": stations},
            "remaining_choices": ["geology-informed layer geometry/thicknesses and rho/vp/vs/Q",
                                  "source position/depth/type/frequency band/duration", "DG degree/timestep and resolution acceptance",
                                  "Rainier receiver/output/checkpoint cadence and field selection"],
            "memory_limit_total_job_bytes": 15_000_000_000,
            "memory_geometry_budget_bytes": guard.budget,
            "cpu_lock_sha256": hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest(),
        }
        with (OUTPUT / "stations.csv").open("w") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "east_local_m", "north_local_m", "elevation_EGM2008_m",
                             "utm_easting_m", "utm_northing_m", "longitude", "latitude", "element_id", "face"])
            for station in stations:
                writer.writerow([station["id"], *station["xyz_m"], *station["utm10n_xy_m"],
                                 *station["lon_lat_degrees"], station["element_id"], station["face"]])
        (OUTPUT / "receivers.json").write_text(json.dumps({
            "simulation_ready": False, "coordinates": metadata["coordinates"],
            "positions": [s["xyz_m"] for s in stations], "station_ids": [s["id"] for s in stations],
        }, indent=2))
        from scripts.rainier_view import write_view
        write_view(OUTPUT, mesh, metadata)
    metadata["geometry_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    metadata["geometry_peak_rss_method"] = "Linux RUSAGE_SELF ru_maxrss KiB*1024, single geometry process; tiny Windows free-memory probe separate"
    metadata["minimum_sampled_windows_available_bytes"] = guard.minimum
    metadata["geometry_wall_seconds"] = time.perf_counter()-started
    (OUTPUT / "model.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
