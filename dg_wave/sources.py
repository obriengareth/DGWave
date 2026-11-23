"""ENU double-couple stress glut and a causal, normalised moment-rate pulse."""

import math

import numpy as np


def double_couple_tensor(strike, dip, rake):
    """Strike clockwise from north; dip right of strike; positive rake is up-dip."""
    if not np.isfinite([strike, dip, rake]).all() or not (0 <= strike < 360 and 0 <= dip <= 90 and -180 <= rake <= 180):
        raise ValueError("double-couple angles require strike[0,360), dip[0,90], rake[-180,180]")
    phi, delta, lam = np.deg2rad([strike, dip, rake])
    along = np.array([np.sin(phi), np.cos(phi), 0.])
    down = np.array([np.cos(delta)*np.cos(phi), -np.cos(delta)*np.sin(phi), -np.sin(delta)])
    normal = np.cross(down, along)
    slip = np.cos(lam)*along-np.sin(lam)*down
    tensor = np.outer(slip, normal)+np.outer(normal, slip)
    tensor[np.abs(tensor) < 32*np.finfo(float).eps] = 0.
    return tensor


def compact_gaussian(nodes, position, width):
    radius = np.linalg.norm(np.asarray(nodes)-position, axis=-1)/width
    u = np.clip(radius-2, 0., 1.)
    taper = 1-u**3*(10+u*(-15+6*u))
    return np.exp(-radius**2/2)*taper


def spatial_normalization(prepared, source):
    space = compact_gaussian(prepared.geometry.nodes, source.position, source.width)
    value = float(np.einsum(
        "ei,i,e->", space, prepared.reference.mass.sum(axis=0), prepared.geometry.determinants,
    ))
    if not np.isfinite(value) or value <= 0:
        raise ValueError("source Gaussian has no positive finite discrete volume integral")
    return value


def moment_rate(time, frequency, delay=0., *, xp=np):
    """Gamma(shape=8, scale=1/(2*pi*f)) density; integral from delay to infinity = 1."""
    rate = 2*math.pi*frequency
    x = xp.maximum((time-delay)*rate, 0.)
    return rate*x**7*xp.exp(-x)/math.factorial(7)


def moment_fraction(time, frequency, delay=0.):
    x = np.maximum((np.asarray(time)-delay)*2*np.pi*frequency, 0.)
    return 1-np.exp(-x)*sum(x**j/math.factorial(j) for j in range(8))


def band_moment_rate_raw(time, source, *, xp=np):
    """Equal-weight cosines INCLUDING both nominal endpoints; C2 time taper."""
    time = xp.asarray(time)
    frequencies = xp.linspace(source.band[0], source.band[1], 64)
    left = xp.clip(time/source.ramp_time, 0., 1.)
    right = xp.clip((source.duration-time)/source.ramp_time, 0., 1.)
    window = (left**3*(10+left*(-15+6*left)))*(right**3*(10+right*(-15+6*right)))
    return window*xp.mean(xp.cos(2*xp.pi*(time[..., None]-source.delay)*frequencies), axis=-1)


def band_temporal_normalization(source, order=128):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    result = 0.
    for low, high in zip((0., source.ramp_time, source.duration-source.ramp_time),
                         (source.ramp_time, source.duration-source.ramp_time, source.duration), strict=True):
        time = low+(nodes+1)*(high-low)/2
        result += float(np.dot(weights, band_moment_rate_raw(time, source))*(high-low)/2)
    if not np.isfinite(result) or result <= 1e-4*source.duration:
        raise ValueError("band moment rate has nonpositive or ill-conditioned net-moment normalisation")
    return result


def moment_source_values(nodes, source):
    tensor = double_couple_tensor(*source.moment_angles)
    components = np.zeros(9)
    components[3:] = tensor[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]
    # sigma_total = sigma_viscoelastic - M(t)*G. Its divergence is -M.grad(G).
    return -source.amplitude/source.normalization*compact_gaussian(
        nodes, source.position, source.width,
    )[..., None]*components
