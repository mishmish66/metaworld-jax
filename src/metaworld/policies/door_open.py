"""Meta-World's scripted policy for door-open-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, door):
    door = door - jnp.array([0.05, 0., 0.])
    return select(
        (jnp.linalg.norm(hand[:2] - door[:2]) > 0.12, door + jnp.array([0.06, 0.02, 0.2])),
        (jnp.abs(hand[2] - door[2]) > 0.04, door + jnp.array([0.06, 0.02, 0.])),
        default=door,
    )


def policy(obs):
    hand, door = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, door), p=25.), 1.)
