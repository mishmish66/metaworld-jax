"""Meta-World's scripted policy for bin-picking-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select

_BIN = jnp.array([0.12, 0.7, 0.02])


def _cube_target(cube):
    """The cube's grasp point, pulled toward the middle of the bin in y so the fingers close inside the bin."""
    cube = cube + jnp.array([0., 0., 0.03])
    return cube.at[1].set(jnp.maximum(0.675, jnp.minimum(cube[1], 0.725)))


def _desired_pos(hand, cube):
    return select(
        (jnp.linalg.norm(hand[:2] - cube[:2]) > 0.02, cube + jnp.array([0., 0., 0.15])),
        (jnp.abs(hand[2] - cube[2]) > 0.01, cube),
        (
            jnp.linalg.norm(hand[:2] - _BIN[:2]) > 0.02,
            jnp.where(hand[2] < 0.15, hand + jnp.array([0., 0., 0.1]), jnp.array([_BIN[0], _BIN[1], 0.18])),
        ),
        default=_BIN,
    )


def _grab_effort(hand, cube):
    far = (jnp.linalg.norm(hand[:2] - cube[:2]) > 0.02) | (jnp.abs(hand[2] - cube[2]) > 0.02)
    return jnp.where(far, -1., 0.6)


def policy(obs):
    hand, cube = obs[:3], _cube_target(obs[4:7])
    return action(move(hand, _desired_pos(hand, cube), p=25.), _grab_effort(hand, cube))
