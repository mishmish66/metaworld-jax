"""Meta-World's scripted policy for handle-pull-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, handle):
    handle = handle + jnp.array([0., -0.04, 0.])
    return select(
        (jnp.linalg.norm(hand[:2] - handle[:2]) > 0.02, handle),
        # Meta-World targets the handle's height on every axis here
        (jnp.abs(hand[2] - handle[2]) > 0.02, jnp.full(3, handle[2])),
        default=handle + jnp.array([0., 0., 0.1]),
    )


def policy(obs):
    hand, handle = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, handle), p=25.), 1.)
