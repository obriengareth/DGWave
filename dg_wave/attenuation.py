"""Passive generalized Maxwell solids, with e^(i*w*t) and Q=Re(M*)/Im(M*).

M(s)=M_R+sum(dM_l*s*tau_l/(1+s*tau_l)), dM_l>=0, M_R>0.
Input K/mu are storage moduli at f_ref, NOT relaxed/unrelaxed moduli or
exact finite-Q phase-speed moduli. Qp refers to K*+4*mu*/3; Qs to mu*.
Branch stress obeys r'_l=dC_l:epsilon' - r_l/tau_l. Total stress uses C_U.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from dg_wave.config import Attenuation, Material
from dg_wave.materials import COEFFICIENTS, element_material


@lru_cache(maxsize=64)
def _fit(inverse_q, low, high, reference, count):
    tau = 1 / (2 * np.pi * np.geomspace(low / 3, high * 3, count))
    if inverse_q == 0:
        return tau, np.zeros(count), 1.0
    frequencies = np.geomspace(low, high, 65)
    x = 2 * np.pi * frequencies[:, None] * tau
    a, b = x*x / (1+x*x), x / (1+x*x)
    xr = 2 * np.pi * reference * tau
    ar = xr*xr / (1+xr*xr)
    matrix = b - inverse_q * (a-ar)
    target = np.full(len(frequencies), inverse_q)
    best, error = None, float("inf")
    # At most five mechanisms: exhaustive active sets give a small exact NNLS,
    # without adding a fitting dependency or allowing negative branch stiffness.
    for size in range(1, count+1):
        for selected in combinations(range(count), size):
            candidate, *_ = np.linalg.lstsq(matrix[:, selected], target, rcond=None)
            if np.any(candidate < 0):
                continue
            weights = np.zeros(count)
            weights[list(selected)] = candidate
            relaxed = 1 - weights @ ar
            residual = np.linalg.norm(matrix @ weights-target)
            if relaxed > 0 and residual < error:
                best, error = (weights, relaxed), residual
    if best is None:
        raise ValueError("requested Q cannot be fitted with positive relaxed/branch moduli")
    return tau, *best


@dataclass(frozen=True)
class Relaxation:
    settings: Attenuation
    reference: Material
    relaxed: Material
    unrelaxed: Material
    tau: np.ndarray
    bulk: np.ndarray
    shear: np.ndarray
    qp: np.ndarray
    qs: np.ndarray

    @property
    def mechanisms(self):
        return len(self.tau)

    @property
    def rate_limit(self):
        return 2 / float(np.min(self.tau))

    def moduli(self, s, *, elements=slice(None)):
        s = np.asarray(s)
        response = s[..., None] * self.tau / (1 + s[..., None] * self.tau)
        return (
            self.relaxed.bulk_modulus[elements, None] + np.einsum("el,fl->ef", self.bulk[elements], response.reshape(-1, self.mechanisms)),
            self.relaxed.shear_modulus[elements, None] + np.einsum("el,fl->ef", self.shear[elements], response.reshape(-1, self.mechanisms)),
        )

    def frequency_chunks(self, s):
        """Bound reporting workspace even when every element has unique coefficients."""
        for start in range(0, len(self.bulk), 128):
            rows = slice(start, start+128)
            bulk, shear = self.moduli(s, elements=rows)
            yield rows, bulk, shear

    def report(self):
        frequencies = np.geomspace(*self.settings.band, 129)
        error_p, error_s = 0.0, 0.0
        for rows, bulk, shear in self.frequency_chunks(2j*np.pi*frequencies):
            p = bulk + 4*shear/3
            qp, qs = p.real/p.imag, shear.real/shear.imag
            # np.maximum retains NaNs so the fit validation cannot hide failures.
            error_p = np.maximum(error_p, np.max(np.abs(qp/self.qp[rows, None]-1)))
            error_s = np.maximum(error_s, np.max(np.abs(qs/self.qs[rows, None]-1)))
        report = {
            "band_hz": list(self.settings.band), "reference_frequency_hz": self.settings.reference_frequency,
            "qp_target": self.settings.qp, "qs_target": self.settings.qs,
            "max_qp_relative_error": float(error_p),
            "max_qs_relative_error": float(error_s),
            "tau_seconds": self.tau.tolist(), "relaxation_dt_bound": 1/self.rate_limit,
            "convention": "input K/mu are storage moduli at reference frequency; Q=storage/loss",
        }
        if self.settings.material_q:
            report["material_q_targets"] = [
                {"tag": tag, "qp": qp, "qs": qs} for tag, qp, qs in self.settings.material_q
            ]
        return report


def _quality_targets(settings, count, physical_tags):
    if (settings.qp is None) != (settings.qs is None):
        raise ValueError("attenuation requires both global Qp/Qs or neither")
    if settings.qp is not None and (
        not np.isfinite([settings.qp, settings.qs]).all() or min(settings.qp, settings.qs) <= 0
    ):
        raise ValueError("attenuation global Qp/Qs must be positive finite")
    qp = np.full(count, np.nan if settings.qp is None else settings.qp, dtype=np.float64)
    qs = np.full(count, np.nan if settings.qs is None else settings.qs, dtype=np.float64)
    if settings.material_q:
        tags = np.asarray(physical_tags)
        if tags.shape != (count,) or not np.issubdtype(tags.dtype, np.integer):
            raise ValueError("material-specific attenuation requires one integer physical tag per element")
        seen = set()
        for tag, p, s in settings.material_q:
            if type(tag) is not int or tag < 1 or tag in seen:
                raise ValueError("attenuation material Q tags must be distinct positive integers")
            if not np.isfinite([p, s]).all() or min(p, s) <= 0:
                raise ValueError("attenuation material Qp/Qs must be positive finite")
            seen.add(tag)
            qp[tags == tag], qs[tags == tag] = p, s
    if not np.isfinite(qp).all() or not np.isfinite(qs).all() or np.any(qp <= 0) or np.any(qs <= 0):
        raise ValueError("attenuation requires positive finite Qp/Qs for every element")
    for values in (qp, qs):
        values.setflags(write=False)
    return qp, qs


def fit_relaxation(
    material: Material, settings: Attenuation | None, *, physical_tags=None,
) -> Relaxation | None:
    if settings is None:
        return None
    low, high = settings.band
    if (
        not np.isfinite([low, high, settings.reference_frequency]).all()
        or not 0 < low < high or not low <= settings.reference_frequency <= high
        or type(settings.mechanisms) is not int
        or not 1 <= settings.mechanisms <= 5
    ):
        raise ValueError("invalid attenuation band/reference/Q/mechanism settings")
    material = element_material(material, max(np.size(getattr(material, name)) for name in COEFFICIENTS))
    qp, qs = _quality_targets(settings, len(material.density), physical_tags)
    k, mu = material.bulk_modulus, material.shear_modulus
    p_loss, shear_loss = (1+4*mu/(3*k))/qp, 4*mu/(3*k*qs)
    inverse_bulk_q = p_loss-shear_loss
    if np.any(inverse_bulk_q < -64*np.finfo(float).eps*np.maximum(p_loss, shear_loss)):
        raise ValueError("incompatible Qp/Qs: requested P loss requires negative bulk dissipation")
    inverse_bulk_q = np.maximum(0, inverse_bulk_q)

    def coefficients(reference_modulus, inverse_q):
        values = np.broadcast_to(inverse_q, np.shape(reference_modulus)).ravel()
        branches, relaxed = [], []
        for value in values:
            tau, weights, residual = _fit(float(value), low, high, settings.reference_frequency, settings.mechanisms)
            branches.append(weights)
            relaxed.append(residual)
        reference_modulus = np.asarray(reference_modulus).reshape(-1)
        return tau, reference_modulus[:, None]*np.array(branches), reference_modulus*np.array(relaxed)

    tau, dk, kr = coefficients(k, inverse_bulk_q)
    _, dm, mur = coefficients(mu, 1/qs)
    relaxed = element_material(Material(material.density, kr, mur), len(kr))
    unrelaxed = element_material(Material(material.density, kr+dk.sum(axis=1), mur+dm.sum(axis=1)), len(kr))
    model = Relaxation(settings, material, relaxed, unrelaxed, tau, dk, dm, qp, qs)
    errors = model.report()
    fit_errors = [errors["max_qp_relative_error"], errors["max_qs_relative_error"]]
    if not np.isfinite(fit_errors).all() or max(fit_errors) > 0.1:
        raise ValueError("passive model Q fit exceeds 10% across band; revise band/mechanism count/Q")
    return model
