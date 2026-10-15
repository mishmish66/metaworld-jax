"""Meta-World's scripted policy for sweep-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, cube, goal):
    cube = cube + jnp.array([0., 0., 0.015])
    left = hand[0] < 0.2
    return select(
        (left & (jnp.linalg.norm(hand[:2] - cube[:2]) > 0.04), cube + jnp.array([0., 0., 0.3])),
        (left & (jnp.abs(hand[2] - cube[2]) > 0.04), cube),
        default=goal + jnp.array([0., 0., 0.1]),
    )


def _grab_effort(hand, cube):
    far = (jnp.linalg.norm(hand[:2] - cube[:2]) > 0.04) | (jnp.abs(hand[2] - cube[2]) > 0.15)
    return select((far, -1.), (cube[0] < 0.4, 0.7), default=-1.)


def policy(obs):
    hand, cube = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, cube, obs[-3:]), p=25.), _grab_effort(hand, cube))
