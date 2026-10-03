"""Meta-World's scripted policy for shelf-place-v3."""
import jax.numpy as jnp

from metaworld.policies.base import action, move, select


def _desired_pos(hand, block, shelf_x):
    block = block + jnp.array([-0.005, 0., 0.015])
    return select(
        (jnp.linalg.norm(hand[:2] - block[:2]) > 0.04, block + jnp.array([0., 0., 0.3])),
        (jnp.abs(hand[2] - block[2]) > 0.04, block),
        (jnp.abs(hand[0] - shelf_x) > 0.02, jnp.stack([shelf_x, hand[1], 0.3])),
        (hand[2] < 0.30, hand + jnp.array([0., 0., 0.30])),
        default=hand + jnp.array([0., 0.05, 0.]),
    )


def _grab_effort(hand, block):
    far = (jnp.linalg.norm(hand[:2] - block[:2]) > 0.04) | (jnp.abs(hand[2] - block[2]) > 0.15)
    return jnp.where(far, -1., 0.7)


def policy(obs):
    hand, block = obs[:3], obs[4:7]
    return action(move(hand, _desired_pos(hand, block, obs[-3]), p=25.), _grab_effort(hand, block))
