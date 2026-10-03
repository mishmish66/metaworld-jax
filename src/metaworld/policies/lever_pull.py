"""Meta-World's scripted policy for lever-pull-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, lever):
    lever = lever + jnp.array([0., -0.055, 0.])
    return select(
        (jnp.linalg.norm(hand[:2] - lever[:2]) > 0.02, lever + jnp.array([0., 0., -0.1])),
        (jnp.abs(hand[2] - lever[2]) > 0.02, lever),
        default=lever + jnp.array([0., 0.08, 0.02]),
    )


def policy(obs):
    hand, lever = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, lever), p=25.), 1.)
