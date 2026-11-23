"""Source-driven serial solves, polynomial receivers, VTU means and an SVG/HTML view."""

import csv
from dataclasses import asdict
from html import escape
import json
import math
import time

import meshio
import modepy as mp
import numpy as np

from dg_wave.config import Config
from dg_wave.checkpoint import chunk_lengths, history_arrays, load_checkpoint, run_target, save_checkpoint
from dg_wave.mesh import read_mesh
from dg_wave.materials import assign_materials, element_material
from dg_wave.operators import _jax_runtime, prepare_mesh
from dg_wave.physics import wave_speeds
from dg_wave.runtime import require_cpu_backend
from dg_wave.solver import ElasticSolver

jax = _jax_runtime()
import jax.numpy as jnp


STATE_NAMES = ("vx", "vy", "vz", "sxx", "syy", "szz", "sxy", "sxz", "syz")


def _source_metadata(source, duration):
    result = asdict(source)
    if source.kind == "double_couple":
        from dg_wave.sources import double_couple_tensor, moment_fraction

        result.update(
            tensor_enu=double_couple_tensor(*source.moment_angles).tolist(),
            amplitude_units="N m, asymptotic scalar moment M0",
            width_units="m, Gaussian sigma; C2 taper between 2 and 3 sigma",
            waveform="causal Gamma(shape8) moment rate with scale 1/(2*pi*frequency); unit time integral to infinity",
            injection="sigma_total' += -M0*tensor_ENU*normalised_G*moment_rate; Pa/s; no external branch-memory injection",
            frequency_meaning="inverse 2*pi*time-scale, NOT a Ricker dominant or hard maximum frequency",
            cumulative_fraction_at_run_end=float(moment_fraction(duration, source.frequency, source.delay or 0.)),
        )
        if source.waveform == "band_moment_rate":
            result.update(
                amplitude_units="N m, NET scalar moment delivered by source.duration",
                waveform="64 equal-weight cosines including band endpoints, phase centred at delay; quintic C2 taper over ramp_time at both ends; normalised NET time integral over duration = 1",
                frequency_meaning="nominal label only; actual frequencies explicitly span source.band including endpoints",
                cumulative_fraction_at_run_end=1. if duration >= source.duration else None,
                spectral_caveat="finite time support/taper causes out-of-band leakage; moment rate can change sign; no strict bandlimited causal pulse claim",
            )
    return result


def resolution_estimate(prepared, solver, band):
    if band is None:
        return None
    points = prepared.mesh.vertices[prepared.mesh.cells]
    edge = float(np.max(np.linalg.norm(points[:, :, None]-points[:, None, :], axis=-1)))
    frequencies = np.geomspace(*band, 129)
    if solver.relaxation is None:
        wavelength = float(np.min(wave_speeds(solver.material)[1]))/band[1]
    else:
        wavelength = float("inf")
        for rows, _, shear in solver.relaxation.frequency_chunks(2j*np.pi*frequencies):
            slowness = np.sqrt(solver.material.density[rows, None]/shear).real
            wavelength = float(np.minimum(wavelength, np.min(1/(frequencies*slowness))))
    return {
        "band_hz": list(band), "max_tetrahedron_edge_m": edge,
        "min_s_wavelength_m": wavelength, "min_wavelength_over_max_edge": wavelength/edge,
        "nominal_degree_scaled_samples_per_wavelength": (prepared.reference.degree+1)*wavelength/edge,
        "interpretation": "conservative geometric proxy, not a resolution/accuracy acceptance criterion; requires mesh/order and boundary studies",
    }


def receiver_interpolation(prepared, positions):
    rows, weights = [], []
    for position in positions:
        delta = np.asarray(position) - prepared.mesh.vertices[prepared.mesh.cells[:, 0]]
        unit = np.einsum("eij,ej->ei", prepared.geometry.inverse_jacobians, delta) / 2
        barycentric = np.column_stack((1 - unit.sum(axis=1), unit))
        inside = np.flatnonzero(np.min(barycentric, axis=1) >= -1e-12)
        if not len(inside):
            raise ValueError(f"receiver {position} lies outside the mesh")
        # A receiver on a DG interface uses the trace from the smallest stable ID.
        element = inside[np.argmin(prepared.mesh.element_ids[inside])]
        reference_point = 2 * unit[element] - 1
        rows.append(element)
        weights.append(mp.resampling_matrix(
            prepared.reference.basis.functions, reference_point[:, None],
            prepared.reference.nodes.T,
        )[0])
    interpolation = np.asarray(weights) if rows else np.empty((0, len(prepared.reference.nodes)))
    return np.asarray(rows, dtype=np.int64), interpolation


def validate_source(prepared, config: Config):
    source = config.source
    if source.amplitude == 0:
        return
    if source.width is None:
        raise ValueError("a nonzero solve source requires source.width in metres")
    lower, upper = prepared.mesh.vertices.min(axis=0), prepared.mesh.vertices.max(axis=0)
    position = np.asarray(source.position)
    if np.any(position < lower) or np.any(position > upper):
        raise ValueError("source.position lies outside the mesh bounding box")
    receiver_interpolation(prepared, [source.position])
    if source.kind == "double_couple":
        from dg_wave.sources import compact_gaussian, spatial_normalization

        if source.moment_angles is None or source.normalization is None:
            raise ValueError("double_couple requires angles and a global spatial normalization")
        integral = spatial_normalization(prepared, source)
        if not np.isclose(integral, source.normalization, rtol=1e-12, atol=0):
            raise ValueError("double_couple normalization does not match this mesh/degree")
        if source.waveform == "band_moment_rate":
            from dg_wave.sources import band_temporal_normalization
            if not np.isclose(band_temporal_normalization(source), source.temporal_normalization, rtol=1e-12, atol=0):
                raise ValueError("double_couple temporal normalization does not match the waveform")
        if np.any(compact_gaussian(
            prepared.geometry.face_nodes[prepared.connectivity.neighbours < 0],
            source.position, source.width,
        ) > 0):
            raise ValueError("double_couple spatial support must not intersect an exterior face")


def smooth_source(prepared, config: Config, *, domain_bounds=None):
    source = config.source
    if source.amplitude == 0:
        return None
    if domain_bounds is None:
        validate_source(prepared, config)
        lower, upper = prepared.mesh.vertices.min(axis=0), prepared.mesh.vertices.max(axis=0)
    else:
        # A partition need not contain the source; root validates it on the global mesh.
        lower, upper = domain_bounds
    if source.width is None:
        raise ValueError("a nonzero solve source requires source.width in metres")
    if source.kind == "double_couple":
        from dg_wave.sources import moment_rate, moment_source_values

        space = jnp.asarray(moment_source_values(prepared.geometry.nodes, source))
        if source.waveform == "band_moment_rate":
            from dg_wave.sources import band_moment_rate_raw
            return lambda t: space*band_moment_rate_raw(t, source, xp=jnp)/source.temporal_normalization
        delay = 0. if source.delay is None else source.delay
        return lambda t: space*moment_rate(t, source.frequency, delay, xp=jnp)
    position = np.asarray(source.position)
    delta = prepared.geometry.nodes - position
    for axis, kind in enumerate(config.boundaries):
        if kind == "periodic":
            length = upper[axis] - lower[axis]
            delta[..., axis] = (delta[..., axis] + length / 2) % length - length / 2
    space = jnp.asarray(
        source.amplitude * np.exp(-np.sum(delta**2, axis=-1) / (2 * source.width**2))
    )[..., None] * jnp.asarray(source.direction)
    delay = 1 / source.frequency if source.delay is None else source.delay

    def envelope(t):
        if source.ramp_time is None:
            return 1.0
        u = jnp.clip(t/source.ramp_time, 0.0, 1.0)
        return u*u*u*(10+u*(-15+6*u))

    if source.band is not None:
        low, high = source.band
        frequencies = low + (high-low)*(np.arange(64)+0.5)/64
        weights = np.sin(np.pi*(np.arange(64)+0.5)/64)**2
        weights *= (frequencies/source.frequency)**2*np.exp(1-(frequencies/source.frequency)**2)
        weights = jnp.asarray(weights/weights.sum())
        frequencies = jnp.asarray(frequencies)

        def band_acceleration(t):
            # A finite Fourier pulse, not a time-compact signal. Starting at t=0
            # truncates its prehistory; resolution studies must include this.
            return space*envelope(t)*jnp.sum(weights*jnp.cos(2*jnp.pi*frequencies*(t-delay)))

        return band_acceleration

    def acceleration(t):
        phase = jnp.pi * source.frequency * (t - delay)
        return space * envelope(t) * (1 - 2 * phase**2) * jnp.exp(-phase**2)

    return acceleration


def _plot_svg(times, series, labels, units):
    values = np.asarray(series)
    low, high = float(np.min(values)), float(np.max(values))
    padding = max((high - low) * 0.08, abs(high) * 0.02, 1e-30)
    low, high = low - padding, high + padding
    if np.min(values) >= 0:
        low = 0.0
    end = max(float(times[-1]), 1e-30)
    colors = ("#0969da", "#cf222e", "#1a7f37")
    parts = ['<svg viewBox="0 0 960 300" role="img" aria-label="Time history">']
    for fraction in np.linspace(0, 1, 6):
        x, y = 90 + 840 * fraction, 240 - 190 * fraction
        parts.append(f'<path d="M90 {y}H930 M{x} 50V240" stroke="#d0d7de" fill="none"/>')
        parts.append(f'<text x="80" y="{y+4}" text-anchor="end">{low+(high-low)*fraction:.3g}</text>')
        parts.append(f'<text x="{x}" y="260" text-anchor="middle">{end*fraction:.3g}</text>')
    for index, (values, label) in enumerate(zip(values, labels, strict=True)):
        points = " ".join(
            f"{90 + 840*t/end:.3f},{240 - 190*(value-low)/(high-low):.3f}"
            for t, value in zip(times, values, strict=True)
        )
        color = colors[index % len(colors)]
        parts.append(f'<polyline points="{points}" stroke="{color}" stroke-width="2" fill="none"/>')
        parts.append(f'<text x="{110+220*index}" y="25" fill="{color}">{escape(label)}</text>')
    parts.append(f'<text x="510" y="290" text-anchor="middle">Time (s)</text><text x="8" y="25">{escape(units)}</text></svg>')
    return "".join(parts)


def _write_view(path, config, times, receivers, energies, summary):
    field_file = summary.get("field_file", "final.vtu")
    attenuating = config.attenuation is not None
    model_name = "viscoelastic" if attenuating else "elastic"
    parts = [
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>DGWave {model_name} box</title>',
        '<style>body{font:16px system-ui;margin:32px auto;max-width:1100px;color:#1f2328;background:#fff}'
        'section{border:1px solid #d0d7de;border-radius:8px;padding:20px;margin:20px 0}svg{width:100%;font-size:13px}'
        'code{background:#f6f8fa;padding:3px}p{line-height:1.6}</style>',
        f'<h1>DGWave - {model_name} box</h1>',
        f'<p>Degree {config.degree}; {summary["elements"]} tetrahedra; '
        f'Maximum P/S speeds {summary["cp"]:.2f} / {summary["cs"]:.2f} m/s; dt {config.dt:g} s. '
        f'Boundary x/y/z: {", ".join(config.boundaries)}.</p>',
        f'Traction-free surface: {config.free_surface}; tags: {config.free_surface_tags}; '
        f'sponge: {config.sponge is not None}.</p>',
        f'<p>Declared source band: {config.source.band} Hz. '
        'Terrain/material accuracy and outer-boundary reflections require separate studies.</p>'
        if config.source.band is not None else '',
        '<p>Local impedance absorbing faces can reflect oblique waves. '
        'An optional artificial sponge damps velocity and stress, not physical attenuation or PML. '
        'Receiver values use polynomial interpolation, choosing the smallest element ID at an interface. '
        f'Open <a href="{escape(field_file)}">{escape(field_file)}</a> '
        'in ParaView for element-mean velocity/stress fields.</p>',
        (
            '<p>Passive generalized Maxwell attenuation is enabled. Input moduli are storage '
            'moduli at the reference frequency; displayed/exported speeds use instantaneous '
            'unrelaxed moduli. Example Q values are synthetic, not measured volcanic-rock data.</p>'
            if attenuating else ''
        ),
        '<section><h2>Discrete stored energy</h2><p>The smooth body source injects energy; '
        'forced energy need not decrease.</p>',
        _plot_svg(times, [energies], ["Kinetic + equilibrium/branch energy" if attenuating else "Kinetic + strain energy"], "J"), '</section>',
    ]
    for receiver, position in enumerate(config.receivers):
        parts.extend([
            f'<section><h2>Receiver {receiver}: {escape(str(position))} m</h2>',
            _plot_svg(times, np.asarray(receivers)[:, receiver, :3].T, ["vx", "vy", "vz"], "m/s"),
            '</section>',
        ])
    parts.append("</html>")
    path.write_text("".join(parts), encoding="utf-8")


def write_histories(config, times, receiver_values, energies, receiver_element_ids):
    with (config.output / "receivers.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "receiver", "element_id", "x_m", "y_m", "z_m", *STATE_NAMES[:config.receiver_components]))
        for index, t in enumerate(times):
            for receiver, position in enumerate(config.receivers):
                writer.writerow((
                    t, receiver, int(receiver_element_ids[receiver]), *position,
                    *receiver_values[index][receiver][:config.receiver_components],
                ))
    with (config.output / "energy.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "energy_J"))
        writer.writerows(zip(times, energies, strict=True))


def write_field(path, prepared, state, global_vertex_ids=None, material=None):
    mean_weights = prepared.reference.mass.sum(axis=0) / (4 / 3)
    means = np.einsum("i,eic->ec", mean_weights, np.asarray(state)[..., :9])
    point_data = {"global_vertex_id": global_vertex_ids} if global_vertex_ids is not None else {}
    cell_data = {
        "element_id": [prepared.mesh.element_ids],
        "mean_velocity": [means[:, :3]], "mean_stress": [means[:, 3:]],
    }
    if material is not None:
        material = element_material(material, len(prepared.mesh.cells))
        cp, cs = wave_speeds(material)
        cell_data.update(density=[material.density], vp=[cp], vs=[cs])
    if prepared.mesh.physical_tags is not None:
        cell_data["material_tag"] = [prepared.mesh.physical_tags]
    meshio.write_points_cells(
        path, prepared.mesh.vertices, [("tetra", prepared.mesh.cells)],
        point_data=point_data,
        cell_data=cell_data,
    )


def run_simulation(config: Config, *, checkpoint=None, restart=None, stop_after=None, extend_run_end=False) -> dict:
    require_cpu_backend(config.backend)
    target = run_target(config, stop_after, checkpoint)
    prepared = prepare_mesh(read_mesh(config.mesh), config.degree)
    material = assign_materials(prepared.mesh, config)
    solver = ElasticSolver(
        prepared, material, config.boundaries, smooth_source(prepared, config),
        free_surface=config.free_surface, sponge=config.sponge,
        attenuation=config.attenuation,
        free_surface_tags=config.free_surface_tags,
    )
    solver.validate_timestep(config.dt)
    rows, weights = receiver_interpolation(prepared, config.receivers)
    state = jnp.zeros(solver.state_shape, dtype=jnp.float64)
    completed = 0
    times, receiver_values, energies = [], [], []
    if restart is not None:
        state, completed, history = load_checkpoint(
            restart, config, prepared.mesh.element_ids, solver.state_shape,
            extend_run_end=extend_run_end,
        )
        state = jnp.asarray(state, dtype=jnp.float64)
        times, receiver_values, energies = (list(history[key]) for key in ("times", "receivers", "energies"))
    if target < completed:
        raise ValueError("requested stop step precedes checkpoint step")
    snapshots = None
    intervals = () if config.surface_interval is None else (config.surface_interval,)
    if intervals:
        from dg_wave.snapshots import SurfaceOutput

        snapshots = SurfaceOutput(
            prepared, solver.free_faces, config, restart_step=completed if restart is not None else None,
        )
    lengths = set(chunk_lengths(completed, target, config.output_interval, *intervals))
    start = time.perf_counter()
    kernels = {
        count: solver.advance_kernel.lower(state, 0.0, config.dt, steps=count).compile()
        for count in lengths
    }
    energy_kernel = solver.energy.lower(state).compile()
    if snapshots is not None:
        snapshots.compile(state)
    compile_seconds = time.perf_counter() - start

    def record(t, current):
        host = np.asarray(current)
        energy = float(energy_kernel(current))
        if not np.isfinite(host).all() or not math.isfinite(energy):
            raise RuntimeError(f"nonfinite solution at time {t:g} s")
        times.append(t)
        receiver_values.append(np.einsum("ri,ric->rc", weights, host[rows, :, :9]))
        energies.append(energy)

    if restart is None:
        record(0.0, state)
        if snapshots is not None:
            snapshots.record(state, 0)
    solve_seconds = 0.0
    for count in chunk_lengths(completed, target, config.output_interval, *intervals):
        start = time.perf_counter()
        state = kernels[count](state, completed * config.dt, config.dt).block_until_ready()
        solve_seconds += time.perf_counter() - start
        completed += count
        if completed % config.output_interval == 0 or completed == config.steps:
            record(completed * config.dt, state)
        if snapshots is not None and completed % config.surface_interval == 0:
            snapshots.record(state, completed)
    if not np.isfinite(np.asarray(state)).all():
        raise RuntimeError("nonfinite solution at checkpoint/final step")
    if checkpoint is not None:
        save_checkpoint(
            checkpoint, config, prepared.mesh.element_ids, state, completed,
            history_arrays(times, receiver_values, energies, len(config.receivers)),
        )
    config.output.mkdir(parents=True, exist_ok=True)
    write_histories(config, times, receiver_values, energies, prepared.mesh.element_ids[rows])
    write_field(config.output / "final.vtu", prepared, state, material=solver.material)
    order = np.argsort(prepared.mesh.element_ids)
    np.savez(
        config.output / "final.npz", element_ids=prepared.mesh.element_ids[order],
        state=np.asarray(state)[order],
    )
    speeds = wave_speeds(solver.material)
    cp, cs = (float(np.max(speed)) for speed in speeds)
    device = next(iter(state.devices()))
    if device.platform != "cpu" or str(state.dtype) != "float64":
        raise RuntimeError("elastic solve requires actual CPU float64 arrays")
    summary = {
        "elements": len(prepared.mesh.cells), "degree": config.degree, "steps": completed,
        "configured_steps": config.steps, "restarted": restart is not None,
        "final_time": completed*config.dt, "cp": cp, "cs": cs, "dt": config.dt,
        "conservative_dt_limit": solver.max_timestep, "boundaries": list(config.boundaries),
        "free_surface": config.free_surface, "sponge": config.sponge is not None,
        "free_surface_tags": list(config.free_surface_tags), "source_band_hz": config.source.band,
        "traction_free_faces": int(np.count_nonzero(solver.free_faces)),
        "resolution_estimate": resolution_estimate(prepared, solver, config.source.band),
        "attenuation": solver.relaxation.report() if solver.relaxation is not None else None,
        "state_components": solver.state_shape[-1], "elastic_reference_dt_limit": solver.elastic_timestep,
        "material_tags": sorted(map(int, np.unique(prepared.mesh.physical_tags))) if prepared.mesh.physical_tags is not None else [],
        "source": _source_metadata(config.source, completed*config.dt),
        "receiver_components": config.receiver_components, "surface_interval_steps": config.surface_interval,
        "cp_range": [float(np.min(speeds[0])), cp], "cs_range": [float(np.min(speeds[1])), cs],
        "backend": device.platform, "dtype": str(state.dtype), "mpi_size": 1,
        "compile_seconds": compile_seconds, "solve_seconds": solve_seconds,
        "timing_method": "compile includes lowering/compilation; solve is synchronised advances only, excluding output I/O",
        "final_energy_J": float(energy_kernel(state)), "peak_energy_J": max(energies),
        "max_receiver_speed": float(np.max(np.linalg.norm(np.asarray(receiver_values)[..., :3], axis=-1))) if len(rows) else 0.0,
        "output": str(config.output),
    }
    _write_view(config.output / "view.html", config, times, receiver_values, energies, summary)
    (config.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
