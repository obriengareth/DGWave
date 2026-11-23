"""Static Rainier run preparation only: no DG residual, compilation or evolution."""

from scripts.rainier_model import JobGuard, ROOT

from dataclasses import replace
import hashlib
import json
import math
import resource
from types import SimpleNamespace

import modepy as mp
import numpy as np

from dg_wave.attenuation import fit_relaxation
from dg_wave.config import Attenuation, Material, Source, load_config
from dg_wave.connectivity import build_connectivity
from dg_wave.geometry import build_geometry
from dg_wave.mesh import TetraMesh
from dg_wave.rainier_layers import effective_materials
from dg_wave.reference import FACE_VERTICES, reference_operators
from dg_wave.sources import (
    band_moment_rate_raw, band_temporal_normalization, compact_gaussian, double_couple_tensor,
    moment_source_values, spatial_normalization,
)


MODEL = ROOT / "output" / "rainier_layered_model"
OUTPUT = ROOT / "output" / "rainier_run_setup"


def boundary_distance(mesh, point):
    rows, faces = np.nonzero(mesh.boundary_tags)
    triangles = mesh.vertices[mesh.cells[rows[:, None], FACE_VERTICES[faces]]]
    a, b, c = np.moveaxis(triangles, 1, 0)
    e, f, delta = b-a, c-a, point-a
    normal = np.cross(e, f)
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    distance = np.sum(delta*normal, axis=1)
    projected = delta-distance[:, None]*normal
    ee, ff, ef = (np.sum(x*y, axis=1) for x, y in ((e, e), (f, f), (e, f)))
    pe, pf = np.sum(projected*e, axis=1), np.sum(projected*f, axis=1)
    u, v = (ff*pe-ef*pf)/(ee*ff-ef*ef), (ee*pf-ef*pe)/(ee*ff-ef*ef)
    result = np.where((u >= 0) & (v >= 0) & (u+v <= 1), np.abs(distance), np.inf)
    for left, right in ((a, b), (a, c), (b, c)):
        edge = right-left
        parameter = np.clip(np.sum((point-left)*edge, axis=1)/np.sum(edge*edge, axis=1), 0, 1)
        result = np.minimum(result, np.linalg.norm(point-left-parameter[:, None]*edge, axis=1))
    index = np.argmin(result)
    return {"minimum_m": float(result[index]), "element_id": int(mesh.element_ids[rows[index]]),
            "face": int(faces[index]), "boundary_tag": int(mesh.boundary_tags[rows[index], faces[index]])}


def main():
    if OUTPUT.exists():
        raise ValueError("refusing to overwrite an existing approval-ready setup")
    protected = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in MODEL.iterdir() if p.is_file()}
    with JobGuard() as guard:
        metadata = json.loads((MODEL / "model.json").read_text())
        with np.load(MODEL / "mesh.npz", allow_pickle=False) as data:
            mesh = TetraMesh(data["vertices"], data["cells"], data["element_ids"],
                             data["material_tags"], data["boundary_tags"])
        ref = reference_operators(2)
        geometry = build_geometry(mesh, ref)
        prepared = SimpleNamespace(mesh=mesh, reference=ref, geometry=geometry)
        table = effective_materials()
        material = Material(*(np.asarray([row[name] for row in table]) for name in
                              ("rho_kg_m3", "bulk_modulus_Pa", "shear_modulus_Pa")))
        attempts = []
        for count in range(1, 6):
            settings = Attenuation(
                (.01, 2.), .5, None, None, count,
                tuple((row["tag"], row["qp"], row["qs"]) for row in table),
            )
            try:
                relaxation = fit_relaxation(material, settings, physical_tags=np.arange(1, 7))
            except ValueError as error:
                if "Q fit exceeds 10%" not in str(error):
                    raise
                attempts.append({"mechanisms": count, "rejected": str(error)})
            else:
                attempts.append({"mechanisms": count, "accepted": True})
                break
        else:
            raise ValueError("requested exploratory band has no acceptable passive Q fit within five mechanisms")
        components = 9+6*relaxation.mechanisms
        relaxed_vs = np.sqrt(relaxation.relaxed.shear_modulus/material.density)
        cp = np.sqrt((relaxation.unrelaxed.bulk_modulus+4*relaxation.unrelaxed.shear_modulus/3)/material.density)
        points = mesh.vertices[mesh.cells]
        edges = np.max(np.stack([np.linalg.norm(points[:, i]-points[:, j], axis=1)
                                 for i in range(4) for j in range(i)]), axis=0)
        layers = []
        for index, row in enumerate(table):
            selection = np.flatnonzero(mesh.physical_tags == row["tag"])
            control = selection[np.argmax(edges[selection])]
            layers.append({
                "tag": row["tag"], "max_edge_m": float(edges[control]),
                "controlling_element_id": int(mesh.element_ids[control]),
                "vertices_m": points[control].tolist(),
                "reference_vs_m_s": row["vs_m_s"], "relaxed_vs_lower_proxy_m_s": float(relaxed_vs[index]),
                "bulk_s_engineering_upper_hz": float(2*relaxed_vs[index]/(10*edges[control])),
            })
        bulk_upper = min(item["bulk_s_engineering_upper_hz"] for item in layers)
        surface_upper = .9*layers[0]["bulk_s_engineering_upper_hz"]
        upper = min(bulk_upper, surface_upper)
        inradius = 3*geometry.volumes/(2*geometry.surface_jacobians.sum(axis=1))
        elastic_dt = .5*inradius/(cp[mesh.physical_tags-1]*9)
        local_dt = elastic_dt/(1+elastic_dt*relaxation.rate_limit)
        controlling = int(np.argmin(local_dt))
        limit = float(local_dt[controlling])
        subdivisions = math.ceil(.004/(.8*limit))
        dt, steps = .004/subdivisions, 1500*subdivisions
        position = (*metadata["dimensions"]["mesh_highest_xy_m"],
                    metadata["dimensions"]["mesh_highest_elevation_m"]-3000.)
        source = Source(position, .5, 1e12, 750., 3.,
                        band=settings.band, ramp_time=.5, kind="double_couple", moment_angles=(0., 90., 90.),
                        waveform="band_moment_rate", duration=6.)
        source = replace(source, temporal_normalization=band_temporal_normalization(source))
        np.testing.assert_allclose(source.temporal_normalization, band_temporal_normalization(source, 256), rtol=1e-13)
        source = replace(source, normalization=spatial_normalization(prepared, source))
        distance = boundary_distance(mesh, np.asarray(position))
        if distance["minimum_m"] <= 3*source.width:
            raise ValueError("proposed compact source intersects the physical boundary")
        unit = np.einsum("eij,ej->ei", geometry.inverse_jacobians, np.asarray(position)-points[:, 0])/2
        bary = np.column_stack((1-unit.sum(axis=1), unit))
        inside = np.flatnonzero(bary.min(axis=1) >= -1e-12)
        if not len(inside) or not np.all(mesh.physical_tags[inside] == 6):
            raise ValueError("expected the 3km-deep source to be inside basement")
        weights = ref.mass.sum(axis=0)
        moment = -np.einsum("eic,i,e->c", moment_source_values(geometry.nodes, source), weights, geometry.determinants)
        np.testing.assert_allclose(moment, source.amplitude*np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]), atol=1e-3, rtol=1e-13)
        # A local ownership split must not independently renormalise the moment.
        from dg_wave.partition import graph_owners

        owners = graph_owners(build_connectivity(mesh, ref), mesh.element_ids, 4)
        split = sum(-np.einsum("eic,i,e->c", moment_source_values(geometry.nodes[part], source),
                               weights, geometry.determinants[part])
                    for part in (np.flatnonzero(owners == rank) for rank in range(4)))
        np.testing.assert_allclose(split, moment, atol=1e-3, rtol=1e-13)
        raw = compact_gaussian(geometry.nodes, position, source.width)
        active = np.flatnonzero(raw.max(axis=1) > 0)
        q = mp.quadrature_for_space(mp.PN(3, 10), mp.Simplex(3))
        near = np.flatnonzero(np.linalg.norm(np.maximum(
            np.maximum(points.min(axis=1)-position, np.asarray(position)-points.max(axis=1)), 0,
        ), axis=1) <= 3*source.width)
        sampled = points[near, :1]+np.einsum("eij,nj->eni", geometry.jacobians[near], q.nodes.T+1)
        integral = float(np.einsum("ei,i,e->", compact_gaussian(sampled, position, source.width),
                                  q.weights, geometry.determinants[near]))
        source_nodes = geometry.nodes[active]
        source_centroid = np.einsum("ei,ij,ejk,e->k", raw[active], ref.mass, source_nodes,
                                    geometry.determinants[active])/source.normalization
        sample_dt, fft_length = 1/2048, 2**21
        sample_time = np.arange(6*2048+1)*sample_dt
        rate = band_moment_rate_raw(sample_time, source)/source.temporal_normalization
        spectrum = np.fft.rfft(rate, n=fft_length)*sample_dt
        frequencies = np.fft.rfftfreq(fft_length, sample_dt)
        power = np.abs(spectrum)**2
        above = frequencies >= upper
        tail_energy = float(np.trapezoid(power[above], frequencies[above])/np.trapezoid(power, frequencies))
        velocity_power = frequencies**2*power
        velocity_tail = float(np.trapezoid(velocity_power[above], frequencies[above])/np.trapezoid(velocity_power, frequencies))
        outside = (frequencies < settings.band[0]) | (frequencies > settings.band[1])
        out_of_band = float(np.trapezoid(np.where(outside, power, 0.), frequencies)/np.trapezoid(power, frequencies))
        receivers = [station["xyz_m"] for station in metadata["stations"]["items"]]
        OUTPUT.mkdir()
        text = f'''[mesh]
path = "../rainier_layered_model/rainier.msh"
[discretization]
degree = 2
[boundary]
types = ["absorbing", "absorbing", "absorbing"]
free_tags = [10]
[attenuation]
band = [0.01, 2.0]
reference_frequency = 0.5
mechanisms = {relaxation.mechanisms}
[source]
kind = "double_couple"
position = {json.dumps(position)}
frequency = {source.frequency!r}
amplitude = {source.amplitude!r}
width = {source.width!r}
delay = 3.0
waveform = "band_moment_rate"
duration = 6.0
ramp_time = 0.5
temporal_normalization = {source.temporal_normalization!r}
strike = 0.0
dip = 90.0
rake = 90.0
normalization = {source.normalization!r}
band = [0.01, 2.0]
[receivers]
positions = {json.dumps(receivers)}
[time]
dt = {dt!r}
steps = {steps}
[runtime]
backend = "cpu"
[output]
directory = "../rainier_run"
interval = {subdivisions}
surface_interval = {25*subdivisions}
receiver_components = 3
'''
        for row in table:
            text += f'\n[materials.{row["tag"]}]\nrho = {row["rho_kg_m3"]}\nvp = {row["vp_m_s"]}\nvs = {row["vs_m_s"]}\n'
            text += f'\n[attenuation.materials.{row["tag"]}]\nqp = {row["qp"]}\nqs = {row["qs"]}\n'
        config_path = OUTPUT / "rainier.toml"
        config_path.write_text(text)
        config = load_config(config_path)
        assert config.steps*config.dt == 6.
        assert config.output_interval*config.dt == .004
        assert np.isclose(config.surface_interval*config.dt, .1, rtol=0, atol=1e-15)
        report = {
            "status": "AWAITING EXPLICIT RUN APPROVAL; USER-SELECTED UNDER-RESOLVED EXPLORATORY CONFIGURATION",
            "run_blocked": False,
            "user_requested_source_band_hz": [.01, 2.],
            "approved": {"depth_below_model_summit_m": 3000, "strike_dip_rake_degrees": [0, 90, 90],
                         "duration_s": 6, "mpi_ranks": 4, "trace_interval_s": .004, "surface_interval_s": .1,
                         "nominal_band_hz": [.01, 2.], "retain_current_underresolved_mesh": True},
            "mesh": {"elements": len(mesh.cells), "degree": 2, "nodes_per_element": len(ref.nodes),
                     "components": components, "scalar_state_dofs": len(mesh.cells)*len(ref.nodes)*components,
                     "one_fp64_state_bytes": len(mesh.cells)*len(ref.nodes)*components*8},
            "frequency_analysis": {
                "criterion": "p*lambda/h_max >= 10 with p=2, i.e. at least FIVE max-edge cells/wavelength; two edge interpolation intervals per p2 element",
                "upper_hz": upper, "bulk_upper_hz": bulk_upper, "surface_proxy_upper_hz": surface_upper,
                "exploratory_warning": "Requested2Hz is NOT resolved by this mesh/order. User explicitly retained mesh and6s duration as under-resolved exploratory configuration. No accuracy claim above this engineering estimate.",
                "window_warning": "0.01Hz has100s period;6s contains0.06cycle and cannot resolve that frequency observationally.",
                "surface_proxy": "0.9*minimum relaxed weathered Vs; engineering margin, not a proven layered/terrain Rayleigh lower bound",
                "layers": layers,
                "limitations": "NOT Nyquist or a validated DG dispersion tolerance; terrain errors, interfaces, surface-wave dispersion, aspect/face orientation and source discretisation can lower usable frequency. No evolution/time-control study performed.",
            },
            "source": {
                "position_local_enu_m": position, "utm_easting_northing_m": metadata["coordinates"]["utm_origin_m"],
                "z_datum": "EGM2008 orthometric m", "summit_elevation_m": metadata["dimensions"]["mesh_highest_elevation_m"],
                "containing_element_ids": mesh.element_ids[inside].tolist(), "material_tag": 6,
                "vp_vs_rho": [5000, 2900, 2700], "tensor_enu_unit": double_couple_tensor(0, 90, 90).tolist(),
                "sign_convention": "strike clockwise from north, dip right-of-strike; positive rake up-dip; normal=down_dip cross strike, slip=cos(rake)*strike-sin(rake)*down_dip; M=M0*(slip outer normal+normal outer slip)",
                "injection": "external stress glut sigma_total=sigma_viscoelastic-M(t)*G; stress rate -Mdot*G Pa/s; no direct force and no direct Maxwell branch injection",
                "proposed_scalar_moment_Nm": source.amplitude, "proposed_mw_equivalent": 2/3*(np.log10(source.amplitude)-9.1),
                "proposed_sigma_m": source.width, "compact_support_radius_m": 3*source.width,
                "width_over_horizontal_spacing": source.width/500,
                "source_cell_max_edge_m": float(edges[inside].max()), "boundary_clearance": distance,
                "normalization_m3": source.normalization, "normalization_method": "exact integral of represented p2 nodal polynomial using existing DG mass; ONE global factor shared by all ranks",
                "unit_moment_partition_algebra": "sum of four disjoint ownership subsets equals global moment without per-rank normalisation",
                "planned_elements_per_rank": np.bincount(owners, minlength=4).tolist(),
                "source_elements_per_planned_rank": np.bincount(owners[active], minlength=4).tolist(),
                "nonzero_source_elements": len(active), "represented_centroid_enu_m": source_centroid.tolist(),
                "high_order_gaussian_integral_m3": integral,
                "represented_vs_high_order_integral_relative_difference": source.normalization/integral-1,
                "waveform": "64 equal-weight cosine frequencies linearly spaced0.01-2Hz INCLUDING endpoints, centred3s; multiply by quintic C2 taper0-.5s and5.5-6s; normalise NET integral over0-6s to1",
                "frequency_parameter_hz": source.frequency, "frequency_parameter_meaning": "label only; not a Ricker dominant or cutoff",
                "frequencies_hz": np.linspace(.01, 2., 64).tolist(),
                "temporal_normalization_seconds": source.temporal_normalization,
                "normalization_order128_vs256": "agrees within1e-13 relative",
                "moment_rate_min_max_per_s": [float(rate.min()), float(rate.max())],
                "negative_rate_warning": "Moment rate changes sign (transient reverse slip); proposed waveform is not a calibrated unilateral earthquake rupture.",
                "total_absolute_moment_variation_over_net_moment": float(np.trapezoid(np.abs(rate), sample_time)),
                "spatial_bandwidth_warning": "sigma750m regularises the moment over multiple cells and attenuates high spatial wavenumbers; not a point source or a guarantee of2Hz radiated amplitude",
                "spectrum": "Finite6s support and C2 taper broaden the nominal cosine band; causal finite pulse is NOT strictly bandlimited. No down-filter to the mesh bound.",
                "fraction_moment_rate_spectral_energy_above_mesh_upper": tail_energy,
                "radiated_velocity_proxy": "homogeneous far-field velocity scales with derivative(moment_rate); additional f factor included, NOT a prediction for the heterogeneous numerical wavefield",
                "fraction_velocity_proxy_energy_above_mesh_upper": velocity_tail,
                "fraction_moment_rate_energy_outside_nominal_band": out_of_band,
                "spectral_measurement": "zero-padded FFT of actual tapered pulse sampled1/2048s; df=0.0009765625Hz; numerical spectral estimate, not propagation/accuracy evidence",
                "fraction_net_moment_delivered_at_6s": float(np.trapezoid(rate, sample_time)),
                "IMPORTANT": "EXPLORATORY UNDER-RESOLVED RUN: substantial requested high-frequency energy is above the mesh estimate;0.01Hz cannot be observed over6s; endpoint taper adds spectral leakage. Source waveform, amplitude, width, reference and boundary treatment remain proposed for final approval.",
            },
            "attenuation": {**relaxation.report(), "selected_for_approval_not_calibrated": True,
                            "mechanism_selection_attempts": attempts,
                            "relaxed_vs_m_s": relaxed_vs.tolist(), "instantaneous_vp_m_s": cp.tolist(),
                            "minimum_relaxed_bulk_Pa": float(relaxation.relaxed.bulk_modulus.min()),
                            "minimum_branch_bulk_Pa": float(relaxation.bulk.min()),
                            "minimum_branch_shear_Pa": float(relaxation.shear.min()),
                            "below_band": "Q targets are not imposed below0.01Hz, including static/DC response; passive model tends to its relaxed moduli"},
            "integration": {"dt_s": dt, "steps": steps, "conservative_dt_limit_s": limit,
                            "controlling_element_id": int(mesh.element_ids[controlling]),
                            "controlling_layer": int(mesh.physical_tags[controlling]),
                            "minimum_inradius_m": float(inradius.min()),
                            "formula": ".5*r_in/(cp_unrelaxed*(p+1)^2), divided by(1+elastic_dt*(relaxation_rate+sponge_rate)); <=80% limit then align exactly to4ms",
                            "sponge": "none selected; sigma_sponge=0. Existing characteristic absorbing sides/base, tag10 free surface. Not a PML/perfect-boundary claim.",
                            "rk": "existing classical explicit RK4"},
            "outputs": {"trace_samples_per_station": 1501, "stations": 10, "components": ["vx", "vy", "vz"],
                        "station_id_by_csv_receiver_index": {str(i): station["id"] for i, station in enumerate(metadata["stations"]["items"])},
                        "snapshot_count": 61, "trace_interval_steps": subdivisions,
                        "surface_interval_steps": 25*subdivisions,
                        "surface": "incremental selected free-face3-component velocity files, static geometry once; no volume gather per snapshot",
                        "checkpoints": "end of job only when launcher passes --checkpoint; no periodic cadence",
                        "volume": "existing final-only full-state output"},
            "safeguards": {"total_job_limit_bytes": 15_000_000_000, "host_reserve_bytes": 2_000_000_000,
                           "mpi": "four distinct physical cores, --nooversubscribe --bind-to core --map-by core; runtime verifies topology and native thread limits",
                           "no_resource_review_gate": True, "no_pilot_or_simulation_performed": True,
                           "per_rank_hard_virtual_address_limit_bytes": 3_000_000_000,
                           "aggregate_rss_abort_threshold_bytes": 14_000_000_000,
                           "monitor": "process-tree aggregate RSS every0.2s; WSL available each poll and Windows queries every5s;2GB host/WSL reserve; launch helper inherits500MB soft address limit",
                           "kernel_aggregate_memory_quota": "not available to this unprivileged session; monitored RSS is not a kernel aggregate quota",
                           "memory_caveat": "Reported state bytes are ONE state only, not a total-job/JIT peak estimate or fit guarantee. Hard per-rank3GB VIRTUAL limits (12GB for4workers) may reject JIT even with low RSS; no pilot performed. No runtime estimate offered"},
            "protected_model_file_sha256": protected,
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
        from scripts.rainier_view import write_view

        view_metadata = {**metadata, "run_source": report["source"], "run_blocked": report["run_blocked"]}
        write_view(OUTPUT, mesh, view_metadata, base_href="../rainier_layered_model/")
        np.savez(OUTPUT/"relaxation-six-materials.npz", tau_s=relaxation.tau,
                 bulk_branches_Pa=relaxation.bulk, shear_branches_Pa=relaxation.shear,
                 relaxed_bulk_Pa=relaxation.relaxed.bulk_modulus, relaxed_shear_Pa=relaxation.relaxed.shear_modulus)
    report["static_setup_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    report["static_setup_minimum_sampled_windows_available_bytes"] = guard.minimum
    for name, digest in protected.items():
        if hashlib.sha256((MODEL/name).read_bytes()).hexdigest() != digest:
            raise ValueError("static preparation modified protected model files")
    (OUTPUT/"approval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: report[key] for key in ("frequency_analysis", "integration", "outputs")}))


if __name__ == "__main__":
    main()
