"""Smooth travelling P/S plane-wave fixtures on periodic boxes."""

import numpy as np

from dg_wave.config import Material
from dg_wave.physics import wave_speeds


def wave_parameters(material: Material, kind: str, mode=(1, 0, 0), lengths=(1.0, 1.0, 1.0)):
    mode = np.asarray(mode)
    lengths = np.asarray(lengths, dtype=float)
    if mode.shape != (3,) or not np.issubdtype(mode.dtype, np.integer) or not np.any(mode):
        raise ValueError("plane-wave mode must be a nonzero integer triple")
    if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("plane-wave lengths must be positive finite triples")
    if kind not in {"p", "s"}:
        raise ValueError("plane-wave kind must be 'p' or 's'")
    wavevector = 2 * np.pi * mode / lengths
    direction = wavevector / np.linalg.norm(wavevector)
    polarization = direction.copy()
    if kind == "s":
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        polarization = axis - np.dot(axis, direction) * direction
        polarization /= np.linalg.norm(polarization)
    speed = wave_speeds(material)[0 if kind == "p" else 1]
    return wavevector, direction, polarization, speed


def plane_wave(points, time: float, material: Material, kind: str, mode=(1, 0, 0),
               lengths=(1.0, 1.0, 1.0), amplitude: float = 1.0):
    k, direction, polarization, speed = wave_parameters(material, kind, mode, lengths)
    phase = np.sin(np.asarray(points) @ k - speed * np.linalg.norm(k) * time)
    mu = material.shear_modulus
    lam = material.bulk_modulus - 2 * mu / 3
    stress = -(amplitude / speed) * (
        lam * np.dot(polarization, direction) * np.eye(3)
        + mu * (np.outer(polarization, direction) + np.outer(direction, polarization))
    )
    components = stress[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    state = np.concatenate((amplitude * polarization, components))
    return phase[..., None] * state
