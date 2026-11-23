"""Isotropic elasticity: (vx, vy, vz, sxx, syy, szz, sxy, sxz, syz), tensile stress."""

import numpy as np

from dg_wave.config import Material
from dg_wave.operators import _jax_runtime

jax = _jax_runtime()
import jax.numpy as jnp


def wave_speeds(material: Material) -> tuple[float, float]:
    return (
        np.sqrt(material.bulk_modulus / material.density + (4 / 3) * material.shear_modulus / material.density),
        np.sqrt(material.shear_modulus / material.density),
    )


def stress_matrix(state):
    s = state[..., 3:]
    return jnp.stack((
        jnp.stack((s[..., 0], s[..., 3], s[..., 4]), axis=-1),
        jnp.stack((s[..., 3], s[..., 1], s[..., 5]), axis=-1),
        jnp.stack((s[..., 4], s[..., 5], s[..., 2]), axis=-1),
    ), axis=-2)


def stress_components(matrix):
    return jnp.stack((
        matrix[..., 0, 0], matrix[..., 1, 1], matrix[..., 2, 2],
        matrix[..., 0, 1], matrix[..., 0, 2], matrix[..., 1, 2],
    ), axis=-1)


def volume_rhs(gradient, material: Material):
    """gradient[..., state component, physical coordinate]."""
    mu = jnp.asarray(material.shear_modulus)
    lam = jnp.asarray(material.bulk_modulus) - 2 * mu / 3
    grad_v = gradient[..., :3, :]
    div_v = jnp.trace(grad_v, axis1=-2, axis2=-1)
    stress_rate = lam[..., None, None] * div_v[..., None, None] * jnp.eye(3) + mu[..., None, None] * (
        grad_v + jnp.swapaxes(grad_v, -1, -2)
    )
    div_stress = jnp.stack((
        gradient[..., 3, 0] + gradient[..., 6, 1] + gradient[..., 7, 2],
        gradient[..., 6, 0] + gradient[..., 4, 1] + gradient[..., 8, 2],
        gradient[..., 7, 0] + gradient[..., 8, 1] + gradient[..., 5, 2],
    ), axis=-1)
    return jnp.concatenate((
        div_stress / jnp.asarray(material.density)[..., None], stress_components(stress_rate),
    ), axis=-1)


def riemann_states(left, right, normal, minus: Material, plus: Material):
    """Exact welded interface traces; both tractions use the *left* outward normal.

    Material coefficients may be scalars or broadcastable element/face fields.
    """
    vm, vp = left[..., :3], right[..., :3]
    tm = jnp.einsum("...ij,...j->...i", stress_matrix(left), normal)
    tp = jnp.einsum("...ij,...j->...i", stress_matrix(right), normal)
    vmn = jnp.sum(vm * normal, axis=-1, keepdims=True)
    vpn = jnp.sum(vp * normal, axis=-1, keepdims=True)
    tmn = jnp.sum(tm * normal, axis=-1, keepdims=True)
    tpn = jnp.sum(tp * normal, axis=-1, keepdims=True)
    cpm, csm = wave_speeds(minus)
    cpp, csp = wave_speeds(plus)

    def traces(vl, vr, tl, tr, zm, zp):
        zm, zp = jnp.asarray(zm)[..., None], jnp.asarray(zp)[..., None]
        velocity = (zm * vl + zp * vr + tr - tl) / (zm + zp)
        traction = (zp * tl + zm * tr + zm * zp * (vr - vl)) / (zm + zp)
        return velocity, traction

    vn, tn = traces(vmn, vpn, tmn, tpn, minus.density * cpm, plus.density * cpp)
    vt, tt = traces(
        vm - vmn * normal, vp - vpn * normal, tm - tmn * normal, tp - tpn * normal,
        minus.density * csm, plus.density * csp,
    )
    return vn * normal + vt, tn * normal + tt


def interface_states(left, right, normal, minus: Material, plus: Material, free_surface=None):
    velocity, traction = riemann_states(left, right, normal, minus, plus)
    if free_surface is not None:
        free_v, free_t = free_surface_states(left, normal, minus)
        mask = free_surface[..., None, None]
        velocity = jnp.where(mask, free_v, velocity)
        traction = jnp.where(mask, free_t, traction)
    return velocity, traction


def constitutive_rate(gradient, bulk, shear):
    mu, bulk = jnp.asarray(shear), jnp.asarray(bulk)
    divergence = jnp.trace(gradient, axis1=-2, axis2=-1)
    return stress_components(
        (bulk-2*mu/3)[..., None, None] * divergence[..., None, None] * jnp.eye(3)
        + mu[..., None, None] * (gradient+jnp.swapaxes(gradient, -1, -2))
    )


def surface_correction(left, right, normal, minus: Material, plus: Material, free_surface=None):
    velocity, traction = interface_states(left, right, normal, minus, plus, free_surface)
    dv = velocity - left[..., :3]
    dt = traction - jnp.einsum("...ij,...j->...i", stress_matrix(left), normal)
    mu = jnp.asarray(minus.shear_modulus)
    lam = jnp.asarray(minus.bulk_modulus) - 2 * mu / 3
    stress_rate = (
        lam[..., None, None] * jnp.sum(dv * normal, axis=-1)[..., None, None] * jnp.eye(3)
        + mu[..., None, None] * (dv[..., :, None] * normal[..., None, :] + normal[..., :, None] * dv[..., None, :])
    )
    return jnp.concatenate((
        dt / jnp.asarray(minus.density)[..., None], stress_components(stress_rate),
    ), axis=-1)


def free_surface_states(left, normal, material: Material):
    """Zero numerical traction; the outgoing characteristic fixes surface velocity."""
    traction = jnp.einsum("...ij,...j->...i", stress_matrix(left), normal)
    normal_traction = jnp.sum(traction * normal, axis=-1, keepdims=True) * normal
    cp, cs = wave_speeds(material)
    zp = jnp.asarray(material.density * cp)[..., None]
    zs = jnp.asarray(material.density * cs)[..., None]
    velocity = left[..., :3] - normal_traction / zp - (traction - normal_traction) / zs
    return velocity, jnp.zeros_like(traction)


def absorbing_states(left, normal, material: Material):
    """Zero incoming characteristics: t* = -Z(n)v*, exact at normal incidence.

    This local impedance condition is not a PML; oblique P/S waves can reflect.
    """
    return riemann_states(left, jnp.zeros_like(left), normal, material, material)
