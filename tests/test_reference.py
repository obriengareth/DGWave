from itertools import product
from math import comb, factorial

import numpy as np
import pytest

from dg_wave.reference import reference_operators


TOL = 1.0e-10


def exponents(degree):
    return [powers for powers in product(range(degree + 1), repeat=3) if sum(powers) <= degree]


def monomials(points, powers):
    return np.column_stack([np.prod(points ** power, axis=1) for power in powers])


def exact_integral(powers):
    # Map x = 2*u-1 from the unit simplex; integrate powers of u analytically.
    result = 0.0
    for i, j, k in product(*(range(power + 1) for power in powers)):
        coefficient = np.prod([
            comb(power, index) * 2**index * (-1)**(power - index)
            for power, index in zip(powers, (i, j, k), strict=True)
        ])
        result += coefficient * factorial(i) * factorial(j) * factorial(k) / factorial(i + j + k + 3)
    return 8.0 * result


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_polynomial_exactness_mass_and_traces(degree):
    ref = reference_operators(degree)
    powers = exponents(degree)
    values = monomials(ref.nodes, powers)
    for axis, derivative in enumerate(ref.differentiation):
        expected = np.zeros_like(values)
        for column, power in enumerate(powers):
            if power[axis]:
                reduced = list(power)
                reduced[axis] -= 1
                expected[:, column] = power[axis] * np.prod(ref.nodes ** reduced, axis=1)
        np.testing.assert_allclose(derivative @ values, expected, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(
        np.einsum("fij,jk->fik", ref.trace, values),
        np.stack([monomials(points, powers) for points in ref.face_nodes]),
        atol=TOL, rtol=TOL,
    )
    exact_mass = np.array([
        [exact_integral(tuple(a + b for a, b in zip(left, right, strict=True)))
         for right in powers]
        for left in powers
    ])
    np.testing.assert_allclose(values.T @ ref.mass @ values, exact_mass, atol=TOL, rtol=TOL)
    np.testing.assert_allclose(ref.mass, ref.mass.T, atol=TOL, rtol=TOL)
    assert np.linalg.eigvalsh(ref.mass).min() > 0


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_lifting_and_integration_by_parts(degree):
    ref = reference_operators(degree)
    expected = np.concatenate([
        ref.trace[face].T @ ref.face_mass[face] for face in range(4)
    ], axis=1)
    np.testing.assert_allclose(ref.mass @ ref.lift, expected, atol=TOL, rtol=TOL)
    for axis, derivative in enumerate(ref.differentiation):
        boundary = sum(
            ref.normals[face, axis] * ref.trace[face].T
            @ ref.face_mass[face] @ ref.trace[face]
            for face in range(4)
        )
        np.testing.assert_allclose(
            ref.mass @ derivative + derivative.T @ ref.mass, boundary,
            atol=TOL, rtol=TOL,
        )
    # A constant vector's outward flux integrates to zero.
    for axis in range(3):
        flux = np.repeat(ref.normals[:, axis], ref.face_nodes.shape[1])
        integral = np.ones(len(ref.nodes)) @ ref.mass @ ref.lift @ flux
        np.testing.assert_allclose(integral, 0.0, atol=TOL, rtol=TOL)
