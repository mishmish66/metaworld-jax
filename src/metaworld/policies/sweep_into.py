"""Meta-World's scripted policy for sweep-into-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, cube, goal):
    cube = cube + jnp.array([-0.005, 0., 0.01])
    return select(
        (jnp.linalg.norm(hand[:2] - cube[:2]) > 0.04, cube + jnp.array([0., 0., 0.3])),
        (jnp.abs(hand[2] - cube[2]) > 0.04, cube),
        default=goal,
    )


def _grab_effort(hand, cube):
    far = (jnp.linalg.norm(hand[:2] - cube[:2]) > 0.04) | (jnp.abs(hand[2] - cube[2]) > 0.15)
    return jnp.where(far, -1., 0.7)


def policy(obs):
    hand, cube = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, cube, obs[-3:]), p=25.), _grab_effort(hand, cube))
