"""Helpers of Meta-World's scripted policies."""
import jax.numpy as jnp


def move(from_xyz, to_xyz, p):
    """Displacement action toward `to_xyz` from `from_xyz` with gain `p`."""
    return p * (to_xyz - from_xyz)


def action(delta_pos, grab_effort):
    """Action of a hand displacement and a grab effort."""
    return jnp.concatenate([delta_pos, jnp.atleast_1d(grab_effort)]).astype(jnp.float32)


def select(*cases, default):
    """The value of the first case whose condition holds, as an if / elif / else chain: cases are (condition, value)."""
    out = default
    for condition, value in reversed(cases):
        out = jnp.where(condition, value, out)
    return out
