"""
Meta-World's reward utilities, written by the authors of dm_control, as JAX functions.
"""
import numpy as np

import jax.numpy as jnp

_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
    """Maps `x` >= 0 to (0, 1], with 1 at 0 and `value_at_1` at 1."""
    match sigmoid:
        case "gaussian":
            scale = np.sqrt(-2. * np.log(value_at_1))
            return jnp.exp(-0.5 * (x * scale) ** 2)
        case "hyperbolic":
            return 1. / jnp.cosh(x * np.arccosh(1. / value_at_1))
        case "long_tail":
            return 1. / ((x * np.sqrt(1. / value_at_1 - 1.)) ** 2 + 1.)
        case "reciprocal":
            return 1. / (jnp.abs(x) * (1. / value_at_1 - 1.) + 1.)
        case "cosine":
            scaled = x * np.arccos(2. * value_at_1 - 1.) / np.pi
            return jnp.where(jnp.abs(scaled) < 1., (1. + jnp.cos(np.pi * scaled)) / 2., 0.)
        case "linear":
            scaled = x * (1. - value_at_1)
            return jnp.where(jnp.abs(scaled) < 1., 1. - scaled, 0.)
        case "quadratic":
            scaled = x * np.sqrt(1. - value_at_1)
            return jnp.where(jnp.abs(scaled) < 1., 1. - scaled ** 2, 0.)
        case "tanh_squared":
            return 1. - jnp.tanh(x * np.arctanh(np.sqrt(1. - value_at_1))) ** 2
    raise ValueError(f"unknown sigmoid {sigmoid!r}")


def tolerance(x, bounds=(0., 0.), margin=0., sigmoid="gaussian", value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
    """1 when `x` is within `bounds`, falling off to `value_at_margin` at `margin` outside them; 0 outside when `margin` is 0."""
    lower, upper = bounds
    in_bounds = (lower <= x) & (x <= upper)
    margin = jnp.asarray(margin)
    safe_margin = jnp.where(margin == 0., 1., margin)
    distance = jnp.where(x < lower, lower - x, x - upper) / safe_margin
    outside = jnp.where(margin == 0., 0., _sigmoids(distance, value_at_margin, sigmoid))
    return jnp.where(in_bounds, 1., outside)


def inverse_tolerance(x, bounds=(0., 0.), margin=0., sigmoid="reciprocal"):
    """0 when `x` is within `bounds`, rising toward 1 outside them."""
    return 1. - tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid, value_at_margin=0.)


def rect_prism_tolerance(curr, zero, one):
    """Product of `curr`'s fractions of the way from corner `zero` to corner `one` inside their prism; 1 outside it."""
    lo, hi = jnp.minimum(zero, one), jnp.maximum(zero, one)
    in_prism = jnp.all((lo <= curr) & (curr <= hi))
    return jnp.where(in_prism, jnp.prod((curr - zero) / (one - zero)), 1.)


def hamacher_product(a, b):
    """The Hamacher t-norm a b / (a + b - a b), 0 when both are 0."""
    denominator = a + b - a * b
    return jnp.where(denominator > 0., a * b / jnp.where(denominator > 0., denominator, 1.), 0.)
