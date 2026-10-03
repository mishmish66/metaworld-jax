"""Meta-World's scripted policy for handle-press-side-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, handle):
    return select(
        (jnp.linalg.norm(hand[:2] - handle[:2]) > 0.02, handle + jnp.array([0., 0., 0.2])),
        default=handle + jnp.array([0., 0., -0.5]),
    )


def policy(obs):
    hand, handle = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, handle), p=25.), 1.)
