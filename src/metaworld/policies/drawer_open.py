"""Meta-World's scripted policy for drawer-open-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def policy(obs):
    hand, drawer = obs[:3], obs[4:7] + jnp.array([0., 0., -0.02])
    above = jnp.linalg.norm(hand[:2] - drawer[:2]) > 0.06
    lowering = jnp.abs(hand[2] - drawer[2]) > 0.04
    delta = select(
        (above, move(hand, drawer + jnp.array([0., 0., 0.3]), p=4.)),
        (lowering, move(hand, drawer, p=4.)),
        default=move(hand, drawer + jnp.array([0., -0.06, 0.]), p=50.),
    )
    return action(delta, -1.)
