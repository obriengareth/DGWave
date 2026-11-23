"""Same-partition FP64 checkpoints; complete.json atomically commits a generation."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import uuid
import zipfile

import numpy as np


_IO_ERRORS = (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile)


def _root_io(comm, operation):
    if comm is None:
        try:
            return operation()
        except _IO_ERRORS as exc:
            raise ValueError(f"checkpoint: {exc}") from exc
    value, error = None, None
    if comm.rank == 0:
        try:
            value = operation()
        except _IO_ERRORS as exc:
            error = f"checkpoint: {exc}"
    error = comm.bcast(error)
    if error is not None:
        raise ValueError(error)
    return comm.bcast(value)


def _local_io(comm, operation):
    if comm is None:
        return _root_io(None, operation)
    value, error = None, None
    try:
        value = operation()
    except _IO_ERRORS as exc:
        error = f"checkpoint rank {comm.rank}: {exc}"
    errors = comm.allgather(error)
    if any(errors):
        raise ValueError("; ".join(item for item in errors if item))
    return value


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")


def _metadata(config):
    values = asdict(config)
    values.pop("output")
    values.pop("mesh")
    if values["source"]["ramp_time"] is None:
        # Preserve compatibility with checkpoints predating the optional ramp.
        values["source"].pop("ramp_time")
    if values["source"]["kind"] == "force":
        for key in ("kind", "moment_angles", "normalization", "waveform", "duration", "temporal_normalization"):
            values["source"].pop(key)
    elif values["source"]["waveform"] is None:
        for key in ("waveform", "duration", "temporal_normalization"):
            values["source"].pop(key)
    if values["surface_interval"] is None:
        values.pop("surface_interval")
    if values["receiver_components"] == 9:
        values.pop("receiver_components")
    if values["attenuation"] is not None and not values["attenuation"]["material_q"]:
        # Global-Q checkpoints predating spatial targets have no material_q key.
        values["attenuation"].pop("material_q")
    values["mesh_sha256"] = _sha(config.mesh)
    return values


def _atomic(path, write):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        write(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_target(config, stop_after, checkpoint):
    if stop_after is None:
        return config.steps
    if type(stop_after) is not int or not 1 <= stop_after <= config.steps or checkpoint is None:
        raise ValueError("--stop-after requires --checkpoint and a step within the configured run")
    return stop_after


def chunk_lengths(start, target, interval, *other_intervals):
    while start < target:
        count = min(*(value-start % value for value in (interval, *other_intervals)), target-start)
        yield count
        start += count


def history_arrays(times, receivers, energies, count):
    return {
        "times": np.asarray(times, dtype=np.float64),
        "receivers": np.asarray(receivers, dtype=np.float64).reshape(len(times), count, 9),
        "energies": np.asarray(energies, dtype=np.float64),
    }


def save_checkpoint(path, config, ids, state, step, history, comm=None):
    path = Path(path)
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)

    def create():
        generation = uuid.uuid4().hex
        (path / generation).mkdir(parents=True, exist_ok=False)
        return generation

    generation = _root_io(comm, create)
    directory = path / generation

    def write_rank():
        host = np.asarray(state)
        if host.dtype != np.float64 or not np.isfinite(host).all():
            raise ValueError("state must be finite float64")
        filename = f"rank{rank:04d}.npz"
        _atomic(directory / filename, lambda stream: np.savez(stream, element_ids=ids, state=host))
        return filename, _sha(directory / filename)

    record = _local_io(comm, write_rank)
    records = [record] if comm is None else comm.gather(record, root=0)

    def commit():
        _atomic(directory / "history.npz", lambda stream: np.savez(stream, **history))
        metadata = _metadata(config)
        manifest = {
            "version": 1, "mpi_size": size, "step": step, "time": step*config.dt,
            "configuration": metadata, "configuration_sha256": hashlib.sha256(_json(metadata)).hexdigest(),
            "state_convention": "vx,vy,vz,sxx,syy,szz,sxy,sxz,syz; then mechanism-major six branch stresses",
            "files": {**dict(records), "history.npz": _sha(directory / "history.npz")},
        }
        _atomic(directory / "manifest.json", lambda stream: stream.write(_json(manifest)))
        pointer = {"version": 1, "generation": generation, "manifest_sha256": _sha(directory / "manifest.json")}
        _atomic(path / "complete.json", lambda stream: stream.write(_json(pointer)))
        return None

    _root_io(comm, commit)


def load_checkpoint(path, config, ids, state_shape, comm=None, *, extend_run_end=False):
    path = Path(path)
    rank, size = (0, 1) if comm is None else (comm.rank, comm.size)

    def read_manifest():
        pointer = json.loads((path / "complete.json").read_text())
        generation = pointer["generation"]
        if pointer["version"] != 1 or not isinstance(generation, str) or len(generation) != 32 or any(
            c not in "0123456789abcdef" for c in generation
        ):
            raise ValueError("invalid checkpoint completion marker")
        directory = path / generation
        if _sha(directory / "manifest.json") != pointer["manifest_sha256"]:
            raise ValueError("checkpoint manifest checksum mismatch")
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["version"] != 1 or manifest["mpi_size"] != size:
            raise ValueError("checkpoint version/process count mismatch; same partition/process count required")
        saved = manifest["configuration"]
        saved_hash = hashlib.sha256(_json(saved)).hexdigest()
        current = _metadata(config)
        if extend_run_end and config.steps >= saved["steps"]:
            current["steps"] = saved["steps"]
        expected = hashlib.sha256(_json(current)).hexdigest()
        if manifest["configuration_sha256"] != expected or saved_hash != expected:
            raise ValueError("checkpoint mesh/config/material metadata mismatch")
        step = manifest["step"]
        if type(step) is not int or not 0 <= step <= min(config.steps, saved["steps"]) or manifest["time"] != step*config.dt:
            raise ValueError("invalid checkpoint step/time")
        if set(manifest["files"]) != {*(f"rank{i:04d}.npz" for i in range(size)), "history.npz"}:
            raise ValueError("checkpoint rank/history file manifest is incomplete")
        return generation, manifest

    generation, manifest = _root_io(comm, read_manifest)
    directory = path / generation

    def read_rank():
        filename = f"rank{rank:04d}.npz"
        if _sha(directory / filename) != manifest["files"][filename]:
            raise ValueError(f"checkpoint checksum mismatch: {filename}")
        with np.load(directory / filename, allow_pickle=False) as arrays:
            state = arrays["state"].copy()
            if not np.array_equal(arrays["element_ids"], ids):
                raise ValueError("checkpoint partition/global element IDs mismatch")
        if state.shape != state_shape or state.dtype != np.float64 or not np.isfinite(state).all():
            raise ValueError("checkpoint full-state shape/dtype/values mismatch")
        return state

    state = _local_io(comm, read_rank)

    def read_history():
        filename = "history.npz"
        if _sha(directory / filename) != manifest["files"][filename]:
            raise ValueError("checkpoint history checksum mismatch")
        with np.load(directory / filename, allow_pickle=False) as arrays:
            history = {key: arrays[key].copy() for key in ("times", "receivers", "energies")}
        steps = list(range(0, manifest["step"]+1, config.output_interval))
        if manifest["step"] == manifest["configuration"]["steps"] and steps[-1] != manifest["step"]:
            steps.append(manifest["step"])
        expected_times = np.asarray(steps)*config.dt
        if (
            history["times"].shape != expected_times.shape
            or not np.array_equal(history["times"], expected_times)
            or history["receivers"].shape != (len(steps), len(config.receivers), 9)
            or history["energies"].shape != (len(steps),)
            or any(not np.isfinite(a).all() for a in history.values())
        ):
            raise ValueError("checkpoint receiver/energy history continuity mismatch")
        return history

    history = _root_io(comm, read_history)
    return state, manifest["step"], history
